// Silero TTS — a local Russian text-to-speech plugin for Astra.
//
// The Rust process is the plugin: it serves the TTS hooks and keeps a long-lived
// python worker alive. The worker owns the Silero model (torch `.pt` package)
// and speaks JSON-lines over stdio — see worker/worker.py.
//
// Voice selection happens on Astra's Voice page: `tts_voices` lists the five
// Silero V5 speakers, and `tts_synthesize` honours the pinned `voice_id`.
// Model / sample rate / fallback voice live in the plugin's Settings.

use anyhow::{anyhow, bail};
use astra_plugin_sdk::prelude::*;
use serde_json::Value as Json;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The five speakers Silero V5 ships for Russian — ids are Silero's own.
/// Display names come from `locales/` (voice.<id>) at runtime.
const VOICES: &[(&str, &str)] = &[
    ("aidar", "male"),
    ("baya", "female"),
    ("kseniya", "female"),
    ("xenia", "female"),
    ("eugene", "male"),
];
const DEFAULT_VOICE: &str = "xenia";

/// Silero bundles arrive only as torch packages (`.pt`); these are the ids.
const MODELS: &[&str] = &["v5_5_ru", "v4_ru"];
const DEFAULT_MODEL: &str = "v5_5_ru";

const SAMPLE_RATES: &[u32] = &[8_000, 24_000, 48_000];
const DEFAULT_SAMPLE_RATE: u32 = 24_000;

/// What the plugin settings deserialize into. The daemon's first payload is
/// `{}`, so every field must carry a default.
#[astra::config]
struct Settings {
    #[serde(default = "default_voice")]
    default_voice: String,
    #[serde(default = "default_model")]
    model: String,
    #[serde(default = "default_sample_rate")]
    sample_rate: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            default_voice: default_voice(),
            model: default_model(),
            sample_rate: default_sample_rate(),
        }
    }
}

fn default_voice() -> String {
    DEFAULT_VOICE.into()
}
fn default_model() -> String {
    DEFAULT_MODEL.into()
}
fn default_sample_rate() -> u32 {
    DEFAULT_SAMPLE_RATE
}

/// The resolved parameters of one synthesis request.
struct SynthesisRequest {
    text: String,
    voice: String,
    model: String,
    sample_rate: u32,
}

/// A long-lived Silero inference process, speaking JSON-lines over stdio.
struct Worker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl Worker {
    /// Pick the first working python (venv first, then PATH) and wait for the
    /// worker's `{"ready":true}` line so a broken runtime fails here, loudly.
    fn spawn(root: &Path) -> anyhow::Result<Worker> {
        if std::env::var_os("SILERO_TTS_DISABLE_WORKER").is_some() {
            bail!("silero worker disabled for testing (SILERO_TTS_DISABLE_WORKER)");
        }
        let script = root.join("worker").join("worker.py");
        let mut last_error = String::new();
        for interp in interpreter_candidates(root) {
            let mut cmd = Command::new(&interp);
            cmd.arg(&script)
                .current_dir(root)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit());
            let mut child = match cmd.spawn() {
                Ok(child) => child,
                Err(e) => {
                    last_error = format!("cannot launch {}: {e}", interp.display());
                    continue;
                }
            };
            let stdin = match child.stdin.take() {
                Some(stdin) => stdin,
                None => {
                    let _ = child.kill();
                    continue;
                }
            };
            let Some(child_stdout) = child.stdout.take() else {
                let _ = child.kill();
                continue;
            };
            let mut stdout = BufReader::new(child_stdout);
            let mut line = String::new();
            match stdout.read_line(&mut line) {
                Ok(0) => {
                    return Err(anyhow!(
                        "Silero worker exited during startup (interpreter: {}). \
                         Install the runtime with setup.ps1 / setup.sh.",
                        interp.display()
                    ));
                }
                Ok(_) => {}
                Err(e) => {
                    return Err(anyhow!("Silero worker died on startup: {e}").context(format!(
                        "interpreter: {}",
                        interp.display()
                    )));
                }
            }
            let ready: Json = serde_json::from_str(line.trim()).unwrap_or(Json::Null);
            if ready.get("ready") != Some(&Json::Bool(true)) {
                return Err(anyhow!("Silero worker did not report ready: {}", line.trim()));
            }
            return Ok(Worker {
                child,
                stdin,
                stdout,
            });
        }
        Err(anyhow!(
            "no usable python interpreter for the Silero worker({}). \
             Run setup.ps1 / setup.sh first.",
            last_error
        ))
    }

    /// One request in, one WAV file + response out. The worker writes the audio
    /// to a temp file and we read it back, so the JSON line stays small.
    fn synthesize(&mut self, req: &SynthesisRequest) -> anyhow::Result<AudioData> {
        let out = temp_wav_path();
        let payload = json!({
            "cmd": "synthesize",
            "text": req.text,
            "speaker": req.voice,
            "model": req.model,
            "sample_rate": req.sample_rate,
            "out": out.to_string_lossy(),
        });
        writeln!(self.stdin, "{payload}")
            .map_err(|e| anyhow!("write to worker: {e}"))?;
        self.stdin
            .flush()
            .map_err(|e| anyhow!("flush to worker: {e}"))?;

        let mut line = String::new();
        let n = self
            .stdout
            .read_line(&mut line)
            .map_err(|e| anyhow!("read from worker: {e}"))?;
        if n == 0 {
            bail!("Silero worker exited during synthesis");
        }
        let resp: Json = serde_json::from_str(line.trim())
            .map_err(|e| anyhow!("bad worker response {:?}: {e}", line.trim()))?;
        if resp.get("ok") != Some(&Json::Bool(true)) {
            let msg = resp
                .get("error")
                .and_then(Json::as_str)
                .unwrap_or("unknown Silero error");
            bail!("silero worker: {msg}");
        }
        let duration_ms = resp
            .get("duration_ms")
            .and_then(Json::as_u64)
            .unwrap_or(0) as u32;
        let mut data = Vec::new();
        std::fs::File::open(&out)
            .and_then(|mut f| f.read_to_end(&mut data))
            .map_err(|e| anyhow!("cannot read worker output {}: {e}", out.display()))?;
        let _ = std::fs::remove_file(&out);
        Ok(AudioData {
            data,
            format: "wav".into(),
            sample_rate: req.sample_rate,
            duration_ms,
        })
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The folder that carries `worker/worker.py` and `.venv`, in preference order.
/// `CARGO_MANIFEST_DIR` is baked in at build time, which is exactly right for a
/// sideloaded local build; the env var and an exe-relative guess cover copies.
fn plugin_root() -> PathBuf {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("SILERO_TTS_ROOT") {
        candidates.push(PathBuf::from(dir));
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    if let Ok(exe) = std::env::current_exe() {
        // The exe usually sits at <root>/target/release/<name>.exe; walk the
        // whole ancestor chain so a copied or bundled layout still finds the
        // folder that carries `worker/worker.py`.
        let mut dir = exe.parent().and_then(Path::parent);
        while let Some(c) = dir {
            candidates.push(c.to_path_buf());
            dir = c.parent();
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd);
    }
    candidates
        .into_iter()
        .find(|c| c.join("worker").join("worker.py").is_file())
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

/// Python interpreters to try, in order: explicit override, the plugin's own
/// venv, then whatever is on PATH. `Command::new` resolves bare names.
fn interpreter_candidates(root: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = Vec::new();
    if let Ok(p) = std::env::var("SILERO_TTS_PYTHON") {
        v.push(PathBuf::from(p));
    }
    #[cfg(windows)]
    v.push(root.join(".venv").join("Scripts").join("python.exe"));
    #[cfg(not(windows))]
    v.push(root.join(".venv").join("bin").join("python"));
    v.push(PathBuf::from("python"));
    #[cfg(not(windows))]
    v.push(PathBuf::from("python3"));
    v
}

fn temp_wav_path() -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("silero-tts-{}-{n}.wav", std::process::id()))
}

fn valid_voice(id: &str) -> bool {
    VOICES.iter().any(|(v, _)| *v == id)
}

/// A voice's display name in the context's language (RU/EN), with an English
/// fallback so an incomplete locale file or a missing ambient context can never
/// leak a raw key. Voice names are resolved in-process: the daemon only renders
/// declared strings, and `tts_voices` receives no context argument.
fn voice_name(id: &str) -> String {
    let english = match id {
        "aidar" => "Aidar",
        "baya" => "Baya",
        "kseniya" => "Kseniya",
        "xenia" => "Kseniya (xenia)",
        "eugene" => "Yevgeny",
        _ => id,
    };
    match try_ctx() {
        Some(ctx) => {
            let key = format!("voice.{id}");
            if ctx.i18n().has(&key) {
                ctx.i18n().t(&key)
            } else {
                english.to_string()
            }
        }
        None => english.to_string(),
    }
}

fn valid_model(id: &str) -> bool {
    MODELS.contains(&id)
}

/// Precedence: a voice pinned on the Voice page wins; otherwise the configured
/// default; otherwise the built-in fallback. Unrecognised config values fall
/// back silently instead of failing the request.
fn resolve(settings: &Settings, pinned: &str) -> SynthesisRequest {
    let voice = if valid_voice(pinned) {
        pinned.to_string()
    } else if valid_voice(&settings.default_voice) {
        settings.default_voice.clone()
    } else {
        DEFAULT_VOICE.into()
    };
    let model = if valid_model(&settings.model) {
        settings.model.clone()
    } else {
        DEFAULT_MODEL.into()
    };
    let sample_rate = if SAMPLE_RATES.contains(&settings.sample_rate) {
        settings.sample_rate
    } else {
        DEFAULT_SAMPLE_RATE
    };
    SynthesisRequest {
        text: String::new(),
        voice,
        model,
        sample_rate,
    }
}

struct SileroTts {
    settings: Mutex<Settings>,
    worker: Arc<Mutex<Option<Worker>>>,
}

impl Default for SileroTts {
    fn default() -> Self {
        Self {
            settings: Mutex::new(Settings::default()),
            worker: Arc::new(Mutex::new(None)),
        }
    }
}

#[astra::plugin]
impl SileroTts {
    /// Settings arrive at start and on every change; the worker picks them up
    /// per request, so a running plugin honours a change instantly.
    #[hook]
    async fn on_config(&self, _ctx: &PluginContext, cfg: Settings) {
        *self.settings.lock().unwrap() = cfg;
    }

    /// Warm the python worker at startup, fire-and-forget. torch import is
    /// seconds, so without this the *first* synthesize would pay cold-start
    /// inside its RPC window. A failed warmup is not fatal: the next
    /// synthesize spawns on demand anyway.
    #[hook]
    async fn on_start(&self, _ctx: &PluginContext) -> anyhow::Result<()> {
        let worker = Arc::clone(&self.worker);
        tokio::task::spawn_blocking(move || {
            let root = plugin_root();
            let mut slot = worker.lock().unwrap();
            if slot.is_none() {
                match Worker::spawn(&root) {
                    Ok(w) => *slot = Some(w),
                    Err(_) => {}
                }
            }
        });
        Ok(())
    }

    /// Kill the worker so a blocking synthesis reader is unblocked and the
    /// runtime drains before the daemon's grace expires. Best effort: if an
    /// op currently holds the slot, leaving it alone is also fine — the
    /// daemon kills the process group at the grace.
    #[hook]
    async fn on_shutdown(&self, _ctx: &PluginContext) {
        if let Ok(mut slot) = self.worker.try_lock() {
            *slot = None;
        }
    }

    /// The five Silero V5 Russian speakers — they appear on Astra's Voice page
    /// ("поле tts") and the user's pick comes back in `req.voice_id`.
    #[hook]
    async fn tts_voices(&self) -> Vec<VoiceInfo> {
        VOICES
            .iter()
            .map(|(id, gender)| {
                VoiceInfo::new(*id, voice_name(id))
                    .with_language("ru")
                    .with_gender(*gender)
            })
            .collect()
    }

    /// Synthesize one utterance through the shared python worker.
    #[hook]
    async fn tts_synthesize(
        &self,
        _ctx: &PluginContext,
        req: TtsRequest,
    ) -> anyhow::Result<AudioData> {
        let TtsRequest {
            text,
            voice_id,
            ..
        } = req;
        let params = {
            let settings = self.settings.lock().unwrap();
            let mut p = resolve(&settings, &voice_id);
            p.text = text;
            p
        };

        let worker = Arc::clone(&self.worker);
        tokio::task::spawn_blocking(move || -> anyhow::Result<AudioData> {
            let root = plugin_root();
            let mut slot = worker.lock().unwrap();
            if slot.is_none() {
                match Worker::spawn(&root) {
                    Ok(w) => *slot = Some(w),
                    Err(e) => return Err(e),
                }
            }
            let worker_mut = slot.as_mut().expect("worker just spawned");
            match worker_mut.synthesize(&params) {
                Ok(audio) => Ok(audio),
                Err(e) => {
                    // A dead worker is a runtime problem, not a config one:
                    // drop it so the next request respawns a fresh process.
                    *slot = None;
                    Err(e)
                }
            }
        })
        .await
        .map_err(|e| anyhow!("silero synthesis task panicked: {e}"))?
    }
}

astra::main!(SileroTts::default());

#[cfg(test)]
mod tests {
    use super::*;
    use astra_plugin_sdk::testing::Harness;

    #[tokio::test]
    async fn exposes_the_five_silero_speakers() {
        let h = Harness::new(SileroTts::default())
            .start()
            .await
            .expect("the plugin started");
        let voices = h.tts_voices().await;
        assert_eq!(voices.len(), 5);
        assert!(voices.iter().all(|v| v.language == "ru"));
        assert_eq!(voices[0].id, "aidar");
    }

    #[tokio::test]
    async fn voice_names_follow_the_context_language() {
        // The ambient context is process-global, so check each language while
        // its own harness is the one installed.
        let en = Harness::new(SileroTts::default())
            .with_language("en")
            .with_ambient_context()
            .start()
            .await
            .expect("the plugin started (en)");
        assert_eq!(en.tts_voices().await[0].name, "Aidar");

        let ru = Harness::new(SileroTts::default())
            .with_language("ru")
            .with_ambient_context()
            .start()
            .await
            .expect("the plugin started (ru)");
        assert_eq!(ru.tts_voices().await[0].name, "Айдар");
        assert_eq!(ru.tts_voices().await[4].name, "Евгений");
    }

    #[test]
    fn resolution_prefers_pinned_then_defaults_then_builtins() {
        let cfg = Settings::default();
        let r = resolve(&cfg, "");
        assert_eq!(r.voice, "xenia");
        assert_eq!(r.model, "v5_5_ru");
        assert_eq!(r.sample_rate, 24_000);

        let cfg = Settings {
            default_voice: "baya".into(),
            model: "v4_ru".into(),
            sample_rate: 48_000,
        };
        let r = resolve(&cfg, "");
        assert_eq!(r.voice, "baya");
        assert_eq!(r.model, "v4_ru");
        assert_eq!(r.sample_rate, 48_000);

        let r = resolve(&cfg, "eugene");
        assert_eq!(r.voice, "eugene");

        let cfg = Settings {
            default_voice: "nope".into(),
            model: "nope".into(),
            sample_rate: 123,
        };
        let r = resolve(&cfg, "");
        assert_eq!(r.voice, "xenia");
        assert_eq!(r.model, "v5_5_ru");
        assert_eq!(r.sample_rate, 24_000);
    }

    #[tokio::test]
    async fn a_config_payload_arrives_and_keeps_the_plugin_healthy() {
        let h = Harness::new(SileroTts::default())
            .with_config(json!({
                "default_voice": "aidar",
                "model": "v4_ru",
                "sample_rate": 8000
            }))
            .start()
            .await
            .expect("the plugin started with settings");
        assert!(h.health().await.0);
    }

    #[tokio::test]
    async fn synthesize_without_runtime_fails_gracefully() {
        // Do not touch the real worker in unit tests — assert only that the
        // error path is descriptive, not a panic.
        unsafe { std::env::set_var("SILERO_TTS_DISABLE_WORKER", "1") }
        let h = Harness::new(SileroTts::default())
            .start()
            .await
            .expect("the plugin started");
        let err = h
            .tts_synthesize(TtsRequest {
                text: "привет".into(),
                voice_id: String::new(),
                speed: 1.0,
                pitch: 1.0,
            })
            .await
            .expect_err("the worker is disabled for this test");
        assert!(err.to_string().contains("worker"), "{err}");
    }
}
