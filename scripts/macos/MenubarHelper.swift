import AppKit
import Foundation

/// 网页控制台的菜单栏图标：打开网页 / 退出。
/// 用法：menubar-helper --url http://127.0.0.1:8790 [--title AI短视频] [--parent-pid N] [--icon /path.png]
@main
enum MenubarHelperMain {
    static func main() {
        let args = Args.parse(CommandLine.arguments)
        guard let url = args.url else {
            fputs("usage: menubar-helper --url URL [--title NAME] [--parent-pid PID] [--icon PATH]\n", stderr)
            exit(2)
        }
        let app = NSApplication.shared
        let delegate = AppDelegate(url: url, title: args.title, parentPid: args.parentPid, iconPath: args.iconPath)
        app.delegate = delegate
        app.setActivationPolicy(.accessory)
        app.run()
    }
}

struct Args {
    var url: URL?
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

final class AppDelegate: NSObject, NSApplicationDelegate {
    let url: URL
    let title: String
    let parentPid: pid_t
    let iconPath: String?
    var statusItem: NSStatusItem?
    var parentWatch: Timer?

    init(url: URL, title: String, parentPid: pid_t, iconPath: String?) {
        self.url = url
        self.title = title
        self.parentPid = parentPid
        self.iconPath = iconPath
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = item.button {
            button.toolTip = "\(title) 控制台"
            if let iconPath, let img = loadTemplateIcon(path: iconPath) {
                button.image = img
                button.imagePosition = .imageLeft
            } else if let symbol = NSImage(
                systemSymbolName: "film",
                accessibilityDescription: title
            ) {
                symbol.isTemplate = true
                button.image = symbol
            } else {
                button.title = "短剧"
            }
        }

        let menu = NSMenu()
        let openItem = NSMenuItem(
            title: "打开网页",
            action: #selector(openWeb(_:)),
            keyEquivalent: "o"
        )
        openItem.target = self
        menu.addItem(openItem)

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

    @objc func openWeb(_ sender: Any?) {
        NSWorkspace.shared.open(url)
    }

    @objc func copyURL(_ sender: Any?) {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(url.absoluteString, forType: .string)
    }

    @objc func quit(_ sender: Any?) {
        if parentPid > 0 {
            kill(parentPid, SIGTERM)
        }
        // 给父进程一点时间收尾；随后自己退出
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
            NSApp.terminate(nil)
        }
    }
}
