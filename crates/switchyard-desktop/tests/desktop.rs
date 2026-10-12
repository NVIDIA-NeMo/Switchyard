// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checks the installer, the server key helper, and the app icons through their public entry
//! points. The installer runs against fake build and service tools, never real ones.

use std::{fs, path::Path, process::Command};
use switchyard_desktop::install::credentials;

#[path = "../examples/desktop-icons/export.rs"]
mod icons;

#[test]
fn committed_icons_match_their_sources_and_a_stale_export_is_reported() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("icons");
    icons::export(&source, true).expect("committed icons are stale; run cargo desktop-icons");
    let copy = tempfile::tempdir().expect("copy");
    for name in "icon.png tray-source.png icon.icns tray.png tray.rgba".split(' ') {
        fs::copy(source.join(name), copy.path().join(name)).expect("icon");
    }
    fs::write(copy.path().join("icon.ico"), "stale").expect("stale icon");
    let error = icons::export(copy.path(), true).expect_err("stale");
    assert!(error.to_string().contains("icon.ico is stale"), "{error}");
    let kept = fs::read(copy.path().join("icon.ico")).expect("kept");
    assert_eq!(kept, b"stale");
}

#[test]
fn saved_keys_replace_inherited_values_and_a_missing_key_stops_the_start() {
    let dir = tempfile::tempdir().expect("fixture");
    let config = dir.path().join("composite.toml");
    let text = r#"schema_version = 1
targets = {}
routes = {}
llm_clients.gateway = { format = "openai_chat", base_url = "https://e.test", api_key_env = "GATEWAY_TOKEN" }
decision_clients.judge = { format = "system_one", endpoint = "https://e.test", api_key_env = "JUDGE_TOKEN", timeout_ms = 1 }
"#;
    fs::write(&config, text).expect("config");
    let names = credentials::variables(&config).expect("names");
    assert_eq!(Vec::from_iter(&names), ["GATEWAY_TOKEN", "JUDGE_TOKEN"]);

    // A stale value from the service manager must not reach the server.
    let mut server = Command::new("server");
    server.env("GATEWAY_TOKEN", "stale-launchd-value");
    let saved = |name: &str| Ok(Some(format!("saved-{name}")));
    credentials::supply(&mut server, &names, saved).expect("supplied");
    let mut env: Vec<_> = server.get_envs().collect();
    env.sort();
    let env = format!("{env:?}");
    let expected = r#"[("GATEWAY_TOKEN", Some("saved-GATEWAY_TOKEN")), ("JUDGE_TOKEN", Some("saved-JUDGE_TOKEN"))]"#;
    assert_eq!(env, expected);

    // A value exported in a shell does not stand in for a missing saved key.
    let mut server = Command::new("server");
    server.env("JUDGE_TOKEN", "exported-in-shell");
    let gateway_only = |name: &str| Ok((name == "GATEWAY_TOKEN").then(|| "saved".into()));
    let error = credentials::supply(&mut server, &names, gateway_only).expect_err("missing");
    assert!(error.contains("No saved key for JUDGE_TOKEN."), "{error}");
    assert!(error.contains("key set JUDGE_TOKEN"), "{error}");
    assert!(!error.contains("GATEWAY"), "{error}");
    let failed = credentials::supply(&mut server, &names, |_| Err("store unavailable".into()));
    assert_eq!(failed, Err("store unavailable".into()));

    // Bad names and empty keys fail before the credential store is opened.
    for name in ["", " ", "A=B", "A\nB"] {
        assert!(credentials::set(name, "key").is_err(), "{name:?}");
        assert!(credentials::import(name).is_err(), "{name:?}");
        assert!(credentials::remove(name).is_err(), "{name:?}");
    }
    let empty = credentials::set("SWITCHYARD_UNUSED_TEST_KEY", " \n");
    assert!(empty.is_err_and(|e| e.ends_with("is empty.")));

    // Import reads one variable from a login shell whose startup files print around it.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let shell = dir.path().join("shell");
        let script = "#!/bin/sh\necho Hi\nexport SY_VALUE=set\n/bin/sh -c \"$4\"\necho Bye\n";
        fs::write(&shell, script).expect("shell");
        fs::set_permissions(&shell, fs::Permissions::from_mode(0o700)).expect("mode");
        let value = credentials::shell_value(&shell, "SY_VALUE");
        assert_eq!(value, Ok("set".into()));
        let error = credentials::shell_value(&shell, "SY_UNSET").expect_err("unset");
        assert!(error.contains("SY_UNSET is not set"), "{error}");
    }
}

#[cfg(unix)]
mod installer {
    use std::{
        collections::BTreeMap,
        ffi::OsString,
        fs,
        io::{Read, Write},
        os::unix::{ffi::OsStringExt, fs::PermissionsExt},
        path::{Path, PathBuf},
        process::{Command, Output},
    };
    use switchyard_desktop::install::{self, Settings};

    const HELPER: &str = env!("CARGO_BIN_EXE_switchyard-desktop-install");
    #[cfg(target_os = "macos")]
    const SERVICES: &str = "Library/LaunchAgents";
    #[cfg(not(target_os = "macos"))]
    const SERVICES: &str = ".config/systemd/user";
    /// Only these service commands may run before an install fails.
    const PREFLIGHT: [&str; 2] = ["launchctl help", "systemctl --user show-environment"];
    /// One script fakes cargo, codesign, id, launchctl, systemctl, and the built binaries. It
    /// logs service commands, and the server's config check fails when HOME has `fail-check`.
    const TOOL: &str = r#"#!/bin/sh
name=$(basename "$0")
case "$name" in launchctl|systemctl) echo "$name $*" >> "$HOME/service_calls" ;; esac
case "$name $1" in
"cargo build")
  while [ "$1" != --target-dir ]; do shift; done
  mkdir -p "$2/release"
  cp "$0" "$2/release/switchyard-server"
  cp "$0" "$2/release/switchyard-desktop"
  cp "$INSTALLER_BIN" "$2/release/switchyard-desktop-install"
  ps -o pgid= -p $$ > "$2/build-process-group" ;;
"id -u") echo 501 ;;
"launchctl print") exit 1 ;;
"switchyard-server --config") [ ! -e "$HOME/fail-check" ] ;;
esac
"#;

    /// Tests rerun this binary to install with the fake tools on PATH; alone, it does nothing.
    #[test]
    fn child_process() {
        if let Some(file) = std::env::var_os("FIXTURE_SETTINGS") {
            let action = std::env::var("FIXTURE_ACTION").expect("action");
            if let Err(error) = run_child(Path::new(&file), &action) {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
    }

    fn run_child(file: &Path, action: &str) -> Result<(), String> {
        if action != "start_update" {
            return install::execute(&Settings::load(file)?, action, false);
        }
        install::start_update(file)?;
        if install::start_update(file).is_ok() {
            return Err("A second update started while the first was starting.".into());
        }
        for _ in 0..1500 {
            let status = install::update_status(file)?;
            match status["state"].as_str() {
                Some("completed") => return Ok(()),
                Some("failed") => return Err(status.to_string()),
                _ => std::thread::sleep(std::time::Duration::from_millis(20)),
            }
        }
        Err("The updater did not finish within 30 seconds.".into())
    }

    /// This function serves a healthy /health endpoint for the rest of the test process.
    fn health_server() -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("health socket");
        let port = listener.local_addr().expect("address").port();
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let _ = stream.read(&mut [0; 1024]);
                let response = "HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}";
                let _ = stream.write_all(response.as_bytes());
            }
        });
        port
    }

    struct Fixture {
        dir: tempfile::TempDir,
        s: Settings,
        file: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().expect("fixture");
            let (home, bin) = (dir.path().join("home"), dir.path().join("bin"));
            // These characters must stay literal in TOML, plist, and systemd arguments.
            let source = dir.path().join("checkout with ' $() & <quotes>");
            for name in ["cargo", "codesign", "id", "launchctl", "systemctl"] {
                write(&bin.join(name), TOOL);
                let mode = fs::Permissions::from_mode(0o755);
                fs::set_permissions(bin.join(name), mode).expect("mode");
            }
            fs::create_dir_all(&home).expect("home");
            fs::create_dir_all(&source).expect("checkout");
            let s = Settings {
                sy_home: home.join("Switchyard ' $(touch unsafe) \\ & <local> % $"),
                codex_home: home.join(".codex"),
                service_dir: home.join(SERVICES),
                app_dir: None,
                home,
                source,
                cargo: bin.join("cargo"),
                port: health_server(),
                profile: "team-dev".into(),
                model: "composite-gpt-6-sol-gpt-6-luna".into(),
            };
            let file = dir.path().join("install.toml");
            let f = Self { dir, s, file };
            f.save();
            f
        }
        fn save(&self) {
            write(&self.file, &toml::to_string(&self.s).expect("settings"));
        }
        fn path(&self) -> OsString {
            let path = std::env::var_os("PATH").expect("PATH");
            let paths = std::env::split_paths(&path);
            std::env::join_paths(std::iter::once(self.dir.path().join("bin")).chain(paths))
                .expect("PATH")
        }
        fn command(&self, program: &Path) -> Command {
            let mut command = Command::new(program);
            command.env("PATH", self.path()).env("HOME", &self.s.home);
            command
        }
        fn run(&self, action: &str) -> Output {
            let mut child = self.command(&std::env::current_exe().expect("test binary"));
            child.args(["--exact", "installer::child_process", "--nocapture"]);
            child
                .env("FIXTURE_SETTINGS", &self.file)
                .env("FIXTURE_ACTION", action);
            child
                .env("INSTALLER_BIN", HELPER)
                .output()
                .expect("installer")
        }
        fn ok(&self, action: &str) {
            let output = self.run(action);
            assert!(output.status.success(), "{action}: {}", stderr(&output));
        }
        fn helper(&self, args: &str, env: Option<(&str, OsString)>) -> Output {
            let mut helper = self.command(Path::new(HELPER));
            helper
                .args(args.split(' '))
                .envs(env)
                .output()
                .expect("helper")
        }
        fn profile(&self) -> PathBuf {
            self.s.codex_home.join("team-dev.config.toml")
        }
        fn codex_files(&self, marker: &str) -> Vec<String> {
            let entries = fs::read_dir(&self.s.codex_home).expect("Codex settings");
            let paths = entries.flatten().map(|entry| entry.path());
            let marked = paths.filter(|path| path.to_string_lossy().contains(marker));
            marked.map(|path| read(&path)).collect()
        }
        /// This function returns every file under HOME but locks, status, and the service log.
        fn files(&self) -> BTreeMap<PathBuf, Vec<u8>> {
            let mut files = BTreeMap::new();
            let mut pending = vec![self.s.home.clone()];
            while let Some(directory) = pending.pop() {
                let entries = fs::read_dir(directory).expect("directory");
                for path in entries.flatten().map(|entry| entry.path()) {
                    let name = path.file_name().expect("name").to_string_lossy();
                    let skip = ["service_calls", "update-status.toml"].contains(&&*name);
                    if path.is_dir() {
                        pending.push(path);
                    } else if !skip && !name.ends_with(".lock") {
                        files.insert(path.clone(), fs::read(&path).expect("file"));
                    }
                }
            }
            files
        }
    }

    fn write(path: &Path, text: &str) {
        fs::create_dir_all(path.parent().expect("parent")).expect("directory");
        fs::write(path, text).expect("write");
    }
    fn read(path: &Path) -> String {
        fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }
    fn stderr(output: &Output) -> String {
        String::from_utf8_lossy(&output.stderr).into_owned()
    }
    fn toml(path: &Path) -> toml::Value {
        toml::from_str(&read(path)).expect("TOML")
    }

    #[test]
    fn install_reinstall_update_and_uninstall_keep_user_data() {
        let mut f = Fixture::new();
        let (sy_home, codex) = (f.s.sy_home.clone(), f.s.codex_home.clone());
        write(&sy_home.join("menubar.toml"), "baseline_model = 'custom'\n");
        write(&sy_home.join("routing.jsonl"), "history");
        write(&codex.join("config.toml"), "user defaults");
        let legacy_job = f.s.service_dir.join("com.nvidia.switchyard.menubar.plist");
        if cfg!(target_os = "macos") {
            write(&legacy_job, "old menu bar job");
        }

        // A first install moves the old menu bar settings and adds the server URL.
        f.ok("install");
        let settings = f.s.desktop_settings();
        let url = |port| format!("server_url = \"http://127.0.0.1:{port}\"\n");
        assert_eq!(
            read(&settings),
            "baseline_model = 'custom'\n".to_owned() + &url(f.s.port)
        );
        assert!(!sy_home.join("menubar.toml").exists());
        let first = read(&f.profile());
        let profile = toml(&f.profile());
        assert_eq!(profile["model"].as_str(), Some(f.s.model.as_str()));
        assert_eq!(profile["model_context_window"].as_integer(), Some(272000));
        assert!(profile.get("approval_policy").is_none());
        if cfg!(target_os = "macos") {
            let contents = f.s.app().join("Contents");
            let dict = |path: PathBuf| plist::from_file::<_, plist::Dictionary>(path);
            let info = dict(contents.join("Info.plist")).expect("Info.plist");
            let executable = info["CFBundleExecutable"].as_string();
            assert_eq!(executable, Some("switchyard-desktop"));
            let icon = info["CFBundleIconFile"].as_string().expect("icon");
            let icon = fs::read(contents.join("Resources").join(icon)).expect("icon");
            assert_eq!(icon, include_bytes!("../icons/icon.icns"));
            let bundled = install::bundle_metadata(&contents.join("MacOS/switchyard-desktop"));
            assert_eq!(Settings::load(&bundled).expect("bundle").sy_home, sy_home);
            // The plist parses only if the installer escaped the < and & in SY_HOME.
            let job = dict(f.s.service_dir.join("com.nvidia.switchyard.server.plist"));
            let job = job.expect("server job");
            let helper = f.s.binary("switchyard-desktop-install");
            assert_eq!(
                job["ProgramArguments"].as_array().expect("args")[0].as_string(),
                helper.to_str()
            );
            assert!(!legacy_job.exists());
        } else {
            let unit = read(&f.s.service_dir.join("switchyard.service"));
            assert!(unit.contains("<local> %% $$/install.toml\""), "{unit}");
        }
        // A path in the settings never runs as a shell command.
        assert!(!f.s.home.join("unsafe").exists());

        // A reinstall on a new port with a route that sets no window backs up the profile,
        // drops the old window, follows the port, and keeps history and configs.
        f.s.port = health_server();
        f.s.model = "codex-auto-review".into();
        f.save();
        f.ok("install");
        assert!(read(&settings).ends_with(&url(f.s.port)));
        let profile = toml(&f.profile());
        assert_eq!(profile["model"].as_str(), Some("codex-auto-review"));
        assert!(profile.get("model_context_window").is_none());
        assert_eq!(f.codex_files("switchyard-backup"), [first]);
        assert_eq!(read(&sy_home.join("routing.jsonl")), "history");
        assert_eq!(read(&codex.join("config.toml")), "user defaults");
        let server_config = include_str!("../../../scripts/config/composite.toml");
        assert_eq!(read(&sy_home.join("composite.toml")), server_config);
        let recorded = Settings::load(&sy_home.join("install.toml")).expect("record");
        assert_eq!((recorded.port, &recorded.model), (f.s.port, &f.s.model));

        // The detached updater keeps the edited profile and app settings byte for byte.
        let edited = "# Personal\nmodel='custom-selection'\nexperimental='kept'\n";
        write(&f.profile(), edited);
        let custom = format!("# Mine\r\n{}baseline_model='x'\r\n", url(f.s.port));
        write(&settings, &custom);
        // An interrupted earlier update must not block this one.
        write(
            &sy_home.join("update-status.toml"),
            "state = 'running'\nupdated_at = 0\n",
        );
        f.ok("start_update");
        assert_eq!(read(&f.profile()), edited);
        assert_eq!(read(&settings), custom);
        let status = install::update_status(&f.file).expect("status");
        assert_eq!(status["state"], "completed");
        // launchd stops the app's whole process group, so the updater must leave it.
        let build = read(&f.s.source.join("target/build-process-group"));
        assert_ne!(build.trim(), unsafe { libc::getpgrp() }.to_string());

        // Uninstall restores the pre-Switchyard Codex defaults, keeps copies of what it
        // replaces, removes the old shell alias, and keeps history.
        write(&codex.join("config.toml"), "current");
        write(&codex.join("config.toml.direct"), "original");
        let alias = "# >>> switchyard codex alias >>>\nalias codex=old\n";
        let rc = f.s.home.join(".bashrc");
        let block = format!("before\n{alias}# <<< switchyard codex alias <<<\nafter\n");
        write(&rc, &block);
        f.ok("uninstall");
        assert_eq!(read(&codex.join("config.toml")), "original");
        assert_eq!(f.codex_files("switchyard-current"), ["current"]);
        assert_eq!(f.codex_files("switchyard-uninstall-recovery"), [edited]);
        assert_eq!(read(&rc), "before\nafter\n");
        assert!(!f.profile().exists() && !f.s.app().exists());
        assert_eq!(read(&sy_home.join("routing.jsonl")), "history");
        f.ok("uninstall");

        // An unfinished alias block is not edited.
        let unfinished = format!("before\n{alias}after\n");
        write(&rc, &unfinished);
        assert!(!f.run("uninstall").status.success());
        assert_eq!(read(&rc), unfinished);
    }

    #[test]
    fn rejected_installs_change_no_files_or_services() {
        let cases = [
            ("install", "check", "--dry-run"),
            ("install", "route", "\"missing-route\""),
            ("install", "profile", "model_providers must be a table"),
            ("install", "url", "server_url value"),
            ("install", "lock", "Could not lock"),
            ("uninstall", "lock", "Could not lock"),
        ];
        for (action, setup, expected) in cases {
            let mut f = Fixture::new();
            let lock = fs::File::create(f.s.home.join(".switchyard-desktop-install.lock"));
            let lock = lock.expect("lock");
            match setup {
                "check" => write(&f.s.home.join("fail-check"), ""),
                "route" => f.s.model = "missing-route".into(),
                "profile" => write(&f.profile(), "model_providers = 'not a table'\n"),
                "url" => write(&f.s.desktop_settings(), "server_url = 42\n"),
                _ => lock.lock().expect("hold lock"),
            }
            f.save();
            let before = f.files();
            let output = f.run(action);
            let error = stderr(&output);
            let rejected = !output.status.success() && error.contains(expected);
            assert!(rejected, "{expected}: {error}");
            assert_eq!(f.files(), before, "{expected}");
            let calls = fs::read_to_string(f.s.home.join("service_calls")).unwrap_or_default();
            let preflight_only = calls.lines().all(|call| PREFLIGHT.contains(&call));
            assert!(preflight_only, "{expected}: {calls}");
        }

        // The helper rejects malformed commands and settings before it writes anything.
        let f = Fixture::new();
        let bad = "key;key set;key status --config;install --dryrun;install --dry-run extra;\
                   uninstall extra;validate --server relative --config /config;\
                   validate --server /server --config /config\n";
        for args in bad.split(';') {
            assert!(!f.helper(args, None).status.success(), "{args:?}");
        }
        let bad = "SY_PROFILE ../escape|;SY_MODEL |a\nb;SY_PORT 0|65536|+1";
        for (name, values) in bad.split(';').filter_map(|case| case.split_once(' ')) {
            let values = values.split('|').map(OsString::from);
            for value in values.chain([OsString::from_vec(vec![0xff])]) {
                let output = f.helper("install", Some((name, value.clone())));
                let error = stderr(&output);
                let rejected = !output.status.success() && error.contains(name);
                assert!(rejected, "{value:?}: {error}");
            }
        }

        // Previews succeed, name the selected installation, and write nothing.
        let previews = "install --dry-run;uninstall --dry-run;\
                        uninstall --settings FILE --dry-run;uninstall --dry-run --settings FILE";
        for args in previews.split(';') {
            let output = f.helper(&args.replace("FILE", &f.file.display().to_string()), None);
            assert!(output.status.success(), "{args}: {}", stderr(&output));
            let names_profile = String::from_utf8_lossy(&output.stdout).contains("team-dev");
            assert_eq!(names_profile, args.contains("FILE"), "{args}");
        }
        assert!(f.files().is_empty());
        write(&f.file, &format!("{}\nunknown = true\n", read(&f.file)));
        assert!(Settings::load(&f.file).is_err());
    }
}

#[cfg(windows)]
#[test]
fn a_locked_windows_executable_rolls_back_every_earlier_replacement() {
    use std::os::windows::fs::OpenOptionsExt;
    use switchyard_desktop::install::{Settings, executable_name, windows};
    let root = tempfile::tempdir().expect("fixture");
    let mut s = Settings::from_env().expect("settings");
    s.source = root.path().join("source");
    s.app_dir = Some(root.path().join("app"));
    let release = s.source.join("target/release");
    fs::create_dir_all(&release).expect("release");
    fs::create_dir_all(s.bin_dir()).expect("app");
    let binaries = "switchyard-server switchyard-desktop switchyard-desktop-install";
    for name in binaries.split(' ') {
        fs::write(release.join(executable_name(name)), "new").expect("new");
        fs::write(s.binary(name), "old").expect("old");
    }
    let mut open = fs::OpenOptions::new();
    let lock = open.read(true).share_mode(0);
    let lock = lock.open(s.binary("switchyard-desktop"));
    assert!(windows::replace_binaries(&s).is_err());
    drop(lock);
    for name in binaries.split(' ') {
        assert_eq!(fs::read(s.binary(name)).expect("old"), b"old");
    }
    windows::replace_binaries(&s).expect("retry");
    for name in binaries.split(' ') {
        assert_eq!(fs::read(s.binary(name)).expect("new"), b"new");
    }
}
