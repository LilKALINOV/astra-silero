# Silero TTS — local Russian text-to-speech for Astra

|<img alt="EN" src="https://img.shields.io/badge/lang-en-green"> English | 
|---| 
On-device Russian voice: text is synthesized **on this machine** with
[Silero TTS](https://github.com/snakers4/silero-models) (V5 by default, V4 as
the fallback). Five Russian speakers: Aidar (m), Baya (f), Kseniya (f),
Kseniya (xenia, f), Yevgeny (m). The model downloads once (~140 MB), then
everything works offline. Audio is returned as WAV at 24 kHz by default.

Process layout: **Rust** is the plugin itself (Astra hooks, worker lifecycle),
**Python** is only the Silero runtime (JSON lines over stdio).

## Requirements

- Windows 10/11 or Linux
- Rust 1.85+ and `protoc` — only to rebuild the binary
- Python 3.9+ (`uv` recommended)

## Install

1. **Set up the environment** (once): `.\setup.ps1` on Windows,
   `./setup.sh` on Linux. Creates `.venv` with PyTorch (CPU).
2. **Build**: `cargo build --release`
3. **Check**: `astra-plugin check . --strict`, then `astra-plugin test .`
4. **Load into Astra**. Developer → Sideload, pick the **folder** of the
   project (`D:\GitHub\astra-silero`), or from the project folder run
   `astra-plugin dev .` (Astra must be running).

## Voice selection

Voices are listed on Astra's voice settings — engine **Silero TTS** (tts
field): Aidar, Baya, Kseniya, Kseniya (xenia), Yevgeny. The chosen voice
arrives as `voice_id` in the request. The plugin settings also carry a
fallback **Default voice** for when no voice is pinned.

Plugin settings (Settings → Silero TTS):

| Field | What it does | Values |
|---|---|---|
| Default voice | Voice used by default | aidar / baya / kseniya / xenia / eugene |
| Model | Model package | v5_5_ru (latest) / v4_ru (fallback) |
| Sample rate (Hz) | Output sample rate | 8000 / 24000 / 48000 |

> Synthesis runs at the original pace: **speed/pitch** are not supported by
> Silero, so those request fields are ignored (always 1.0).

## Localization

All UI strings are translated into **Russian and English**: the store-card
text (`listing.*`), the Settings labels and dropdown options, and the voice
names. They live in `locales/en.json` (base, mandatory) and `locales/ru.json`;
the file layout is Rendered by Astra according to its UI language and re-read
at runtime, so switching language updates voice names without a restart.
Validate with `astra-plugin locale check` (or `astra-plugin check . --strict`).

## First run

On startup the plugin eagerly brings up the python worker in the background, so
torch's slow import never lands inside a synthesis request. The worker also
prefetches the model file `v5_5_ru.pt` (~140 MB) into `.models/` next to the
plugin; if a request arrives first, it downloads the model itself. Afterwards
synthesis is fully offline.

Optional environment variables:

- `SILERO_TTS_ROOT` — plugin root (if the binary was copied elsewhere)
- `SILERO_TTS_PYTHON` — python interpreter (default: `.venv`, then `python`)
- `SILERO_CACHE_DIR` — model cache dir (default: `.models/` at the root)

---

# Silero TTS — локальный русский синтез речи для Astra

|<img alt="RU" src="https://img.shields.io/badge/lang-ru-green"> Русский |
|---|
Плагин голоса: озвучивает текст **на этом же компьютере** движком
[Silero TTS](https://github.com/snakers4/silero-models) (версия V5; V4 как
запасной вариант). Русские голоса — пять: Айдар (м), Бая (ж), Ксения (ж),
Ксения (xenia, ж), Евгений (м). Модель скачивается один раз (~140 МБ) и
потом всё работает офлайн. Аудио отдаётся в WAV, 24 кГц по умолчанию.

Структура процесса: **Rust** — это сам плагин (хуки Astra, жизненный цикл
воркера), **Python** — только среда исполнения модели Silero (JSON-строки
через stdio).

## Требования

- Windows 10/11 или Linux
- Rust 1.85+ и `protoc` — только чтобы пересобрать бинарь
- Python 3.9+ (рекомендуется `uv`)

## Установка

1. **Настройте окружение** (один раз): `.\setup.ps1` на Windows,
   `./setup.sh` на Linux. Создаст `.venv` с PyTorch (CPU).
2. **Соберите**: `cargo build --release`
3. **Проверьте**: `astra-plugin check . --strict`, затем `astra-plugin test .`
4. **Загрузите в Astra**. Разработчик → Sideload, выберите **папку** проекта
   (`D:\GitHub\astra-silero`), либо в консоли из папки проекта:
   `astra-plugin dev .` (Astra должна быть запущена).

## Выбор голоса

Голоса перечисляются в настройках голоса Astra — движок **Silero TTS**
(поле tts): Айдар, Бая, Ксения, Ксения (xenia), Евгений. Выбранный голос
прилетает в запрос как `voice_id` и используется при синтезе. Плюс в
настройках самого плагина есть запасной **Default voice** — на случай, если
голос нигде не выбран.

В настройках плагина (Settings → Silero TTS) доступны:

| Поле | Что делает | Значения |
|---|---|---|
| Default voice | Голос по умолчанию | aidar / baya / kseniya / xenia / eugene |
| Model | Пакет модели | v5_5_ru (последний) / v4_ru (запасной) |
| Sample rate (Hz) | Частота дискретизации | 8000 / 24000 / 48000 |

> Синтез идёт в исходном темпе: **speed/pitch** движок Silero менять не умеет,
> поэтому значения из запроса Astra игнорируются (всегда 1.0).

## Локализация

Все строки интерфейса переведены на **русский и английский**: текст карточки
в магазине (`listing.*`), подписи и опции настроек, имена голосов. Они лежат
в `locales/en.json` (базовая, обязательная) и `locales/ru.json`; подписи
рендерит Astra по языку интерфейса, а имена голосов плагин читает в рантайме
и меняет сразу при переключении языка. Проверка:
`astra-plugin locale check` (или `astra-plugin check . --strict`).

## Первый запуск

При старте плагин **в фоне** поднимает python-воркер, чтобы медленный импорт
torch не попадал внутрь запроса синтеза. Воркер также предзагружает файл
модели `v5_5_ru.pt` (~140 МБ) в папку `.models/` рядом с плагином; если
запрос пришёл раньше — модель докачается сама. Дальше синтез полностью
офлайн.

Переменные окружения (необязательно):

- `SILERO_TTS_ROOT` — корень плагина (если бинарь скопирован в другое место)
- `SILERO_TTS_PYTHON` — путь к интерпретатору (по умолчанию: `.venv`, затем `python`)
- `SILERO_CACHE_DIR` — куда класть модели (по умолчанию: `.models/` в корне)