import AppKit
import Foundation

/// 网页控制台的菜单栏图标：打开网页 / 退出；生成中显示实时进度%。
/// 用法：menubar-helper --url http://127.0.0.1:8790 [--status-url …/api/menubar] [--title AI短视频] [--parent-pid N] [--icon /path.png]
@main
enum MenubarHelperMain {
    static func main() {
        let args = Args.parse(CommandLine.arguments)
        guard let url = args.url else {
            fputs("usage: menubar-helper --url URL [--status-url URL] [--title NAME] [--parent-pid PID] [--icon PATH]\n", stderr)
            exit(2)
        }
        let app = NSApplication.shared
        let delegate = AppDelegate(
            url: url,
            statusURL: args.statusURL ?? url.appendingPathComponent("api/menubar"),
            title: args.title,
            parentPid: args.parentPid,
            iconPath: args.iconPath
        )
        app.delegate = delegate
        app.setActivationPolicy(.accessory)
        app.run()
    }
}

struct Args {
    var url: URL?
    var statusURL: URL?
    var title: String = "AI短视频"
    var parentPid: pid_t = 0
    var iconPath: String?

    static func parse(_ argv: [String]) -> Args {
        var out = Args()
        var i = 1
        while i < argv.count {
            let a = argv[i]
            func next() -> String? {
                i += 1
                return i < argv.count ? argv[i] : nil
            }
            switch a {
            case "--url":
                if let s = next(), let u = URL(string: s) { out.url = u }
            case "--status-url":
                if let s = next(), let u = URL(string: s) { out.statusURL = u }
            case "--title":
                if let s = next(), !s.isEmpty { out.title = s }
            case "--parent-pid":
                if let s = next(), let n = Int32(s) { out.parentPid = n }
            case "--icon":
                if let s = next(), !s.isEmpty { out.iconPath = s }
            default:
                break
            }
            i += 1
        }
        return out
    }
}

struct MenubarStatus: Decodable {
    var running: Bool?
    var pct: Double?
    var project: String?
    var current: String?
    var detail: String?
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    let baseURL: URL
    let statusURL: URL
    let title: String
    let parentPid: pid_t
    let iconPath: String?
    var statusItem: NSStatusItem?
    var parentWatch: Timer?
    var progressWatch: Timer?
    var idleImage: NSImage?
    var lastTitle: String = "\u{0}" // sentinel ≠ ""
    var lastProject: String?
    var openMenuItem: NSMenuItem?

    init(url: URL, statusURL: URL, title: String, parentPid: pid_t, iconPath: String?) {
        self.baseURL = url
        self.statusURL = statusURL
        self.title = title
        self.parentPid = parentPid
        self.iconPath = iconPath
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        idleImage = makeIdleImage()
        if let button = item.button {
            button.toolTip = "\(title) 控制台"
            button.image = idleImage
            button.imagePosition = .imageLeft
            button.title = ""
        }

        let menu = NSMenu()
        let openItem = NSMenuItem(
            title: "打开网页",
            action: #selector(openWeb(_:)),
            keyEquivalent: "o"
        )
        openItem.target = self
        menu.addItem(openItem)
        openMenuItem = openItem

        let copyItem = NSMenuItem(
            title: "复制地址",
            action: #selector(copyURL(_:)),
            keyEquivalent: "c"
        )
        copyItem.target = self
        menu.addItem(copyItem)

        menu.addItem(NSMenuItem.separator())

        let quitItem = NSMenuItem(
            title: "退出",
            action: #selector(quit(_:)),
            keyEquivalent: "q"
        )
        quitItem.target = self
        menu.addItem(quitItem)

        item.menu = menu
        statusItem = item

        if parentPid > 0 {
            parentWatch = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
                self?.checkParent()
            }
        }
        progressWatch = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            self?.pollProgress()
        }
        // 立刻拉一次，别等半秒
        pollProgress()
    }

    private func makeIdleImage() -> NSImage? {
        if let iconPath, let img = loadTemplateIcon(path: iconPath) {
            return img
        }
        if let symbol = NSImage(systemSymbolName: "film", accessibilityDescription: title) {
            symbol.isTemplate = true
            return symbol
        }
        return nil
    }

    private func loadTemplateIcon(path: String) -> NSImage? {
        guard let img = NSImage(contentsOfFile: path) else { return nil }
        let size = NSSize(width: 18, height: 18)
        let resized = NSImage(size: size)
        resized.lockFocus()
        img.draw(
            in: NSRect(origin: .zero, size: size),
            from: NSRect(origin: .zero, size: img.size),
            operation: .copy,
            fraction: 1.0
        )
        resized.unlockFocus()
        resized.isTemplate = true
        return resized
    }

    private func checkParent() {
        guard parentPid > 0 else { return }
        if kill(parentPid, 0) != 0 {
            NSApp.terminate(nil)
        }
    }

    private func pollProgress() {
        var req = URLRequest(url: statusURL)
        req.timeoutInterval = 1.5
        req.cachePolicy = .reloadIgnoringLocalCacheData
        URLSession.shared.dataTask(with: req) { [weak self] data, _, _ in
            guard let self, let data else { return }
            let status = try? JSONDecoder().decode(MenubarStatus.self, from: data)
            DispatchQueue.main.async {
                self.applyStatus(status)
            }
        }.resume()
    }

    private func applyStatus(_ status: MenubarStatus?) {
        let running = status?.running == true
        let project = (status?.project ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        lastProject = project.isEmpty ? nil : project

        let titleText: String
        var tip = "\(title) 控制台"
        if running {
            let pct = status?.pct ?? 0
            let shown = Int(pct.rounded())
            titleText = "\(shown)%"
            if !project.isEmpty {
                tip += " · \(project)"
            }
            if let cur = status?.current, !cur.isEmpty {
                tip += " · \(cur)"
            }
            tip += " · \(shown)%"
        } else {
            titleText = ""
        }

        if titleText != lastTitle, let button = statusItem?.button {
            lastTitle = titleText
            button.title = titleText
            button.image = idleImage
            button.imagePosition = titleText.isEmpty ? .imageOnly : .imageLeft
            button.toolTip = tip
        } else if let button = statusItem?.button {
            button.toolTip = tip
        }

        if let open = openMenuItem {
            if running, !project.isEmpty {
                open.title = "打开网页（\(project)）"
            } else {
                open.title = "打开网页"
            }
        }
    }

    private func openURL() -> URL {
        guard let project = lastProject, !project.isEmpty else {
            return baseURL
        }
        var comps = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)
        var items = comps?.queryItems ?? []
        items.removeAll { $0.name == "project" }
        items.append(URLQueryItem(name: "project", value: project))
        comps?.queryItems = items
        return comps?.url ?? baseURL
    }

    @objc func openWeb(_ sender: Any?) {
        NSWorkspace.shared.open(openURL())
    }

    @objc func copyURL(_ sender: Any?) {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(openURL().absoluteString, forType: .string)
    }

    @objc func quit(_ sender: Any?) {
        if parentPid > 0 {
            kill(parentPid, SIGTERM)
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
            NSApp.terminate(nil)
        }
    }
}
