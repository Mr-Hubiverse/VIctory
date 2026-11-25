//! Accessibility: speech synthesis for the title's narration. Text and SSML
//! are spoken by a host voice (Windows speech synthesis) into a WAVE file of
//! 16 kHz mono 16-bit PCM the title reads back; where the host has no working
//! voice, the stream is valid WAVE audio of no length and narration is
//! silent. The host's voices are listed to the title, which may pick one for
//! a synthesizer.
//!
//! One thread owns the host's synthesizer and speaks one line at a time:
//! WinRT synthesizers made, used and dropped on several title threads at
//! once end the process (`0xc0000409`). A line the host has not spoken
//! within `PATIENCE` is silence.
//!
//! Synthesizer and stream handles are keys into a table, not addresses: a
//! handle the title closed twice, or never had, is refused.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::xgameruntime::{
    Guid, HRESULT, Interface, Methods, Object, S_OK, add_ref, missing, put, text,
};

/// Closed captions, high contrast, speech to text and synthesis.
pub const PROVIDER: Guid = Guid::new(
    0x3e24_1977,
    0x6237,
    0x41e9,
    [0x85, 0x59, 0x10, 0x2c, 0x2d, 0x99, 0x83, 0xf1],
);
/// Unchanged since 2020.
pub const IID: Guid = Guid::new(
    0xd722_b373,
    0x8c4d,
    0x4692,
    [0x9e, 0x51, 0xd6, 0xad, 0x9b, 0x37, 0xaa, 0x7d],
);

pub static OBJECT: Object = Object::new("accessibility", &VTABLE);
pub const INTERFACES: &[Interface] = &[(PROVIDER, IID, &OBJECT)];

/// A WAVE file of 16 kHz mono 16-bit PCM with no samples: RIFF size 36,
/// a 16-byte format chunk (PCM, 1 channel, 16,000 Hz, 32,000 bytes/s,
/// block 2, 16 bits), an empty data chunk.
const SILENCE: [u8; 44] =
    *b"RIFF\x24\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80\x3e\0\0\0\x7d\0\0\x02\0\x10\0data\0\0\0\0";
/// The rate of every stream, as `SILENCE` has it.
const RATE: u32 = 16_000;
/// How long the title waits for the host to speak a line.
const PATIENCE: Duration = Duration::from_secs(10);

/// What a handle the title holds stands for.
enum Entry {
    /// A synthesizer: the voice the title picked, or none for the default.
    Synthesizer(Option<String>),
    /// A stream: its WAVE file.
    Stream(Vec<u8>),
}

/// Open handles. Keys are never reused, so a stale handle stays invalid.
static HANDLES: Mutex<Option<HashMap<usize, Entry>>> = Mutex::new(None);

/// Runs `f` on the table of open handles.
fn handles<T>(f: impl FnOnce(&mut HashMap<usize, Entry>) -> T) -> T {
    let mut table = HANDLES.lock().unwrap_or_else(|e| e.into_inner());
    f(table.get_or_insert_with(HashMap::new))
}

/// A new handle for `entry`: an aligned, non-null value the title only
/// passes back.
fn open(entry: Entry) -> *const u8 {
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    let key = 0x5_0000 + NEXT.fetch_add(1, Ordering::Relaxed) * 16;
    handles(|table| table.insert(key, entry));
    key as *const u8
}

/// The voice of synthesizer `handle`, or `None` if it is not one.
fn voice_of(handle: *const u8) -> Option<Option<String>> {
    handles(|table| match table.get(&(handle as usize)) {
        Some(Entry::Synthesizer(voice)) => Some(voice.clone()),
        _ => None,
    })
}

/// Sets the voice of synthesizer `handle`; false if it is not one.
fn set_voice(handle: *const u8, voice: Option<String>) -> bool {
    handles(|table| match table.get_mut(&(handle as usize)) {
        Some(Entry::Synthesizer(picked)) => {
            *picked = voice;
            true
        }
        _ => false,
    })
}

/// The length of stream `handle`'s WAVE file, or `None` if it is not one.
fn stream_len(handle: *const u8) -> Option<usize> {
    handles(|table| match table.get(&(handle as usize)) {
        Some(Entry::Stream(wave)) => Some(wave.len()),
        _ => None,
    })
}

/// Closes `handle` if it is a synthesizer (`stream` false) or a stream.
fn close_handle(handle: *const u8, stream: bool) -> HRESULT {
    handles(|table| {
        let key = handle as usize;
        let kind = match table.get(&key) {
            Some(Entry::Stream(_)) => true,
            Some(Entry::Synthesizer(_)) => false,
            None => return super::xasync::E_INVALIDARG,
        };
        if kind != stream {
            return super::xasync::E_INVALIDARG;
        }
        table.remove(&key);
        S_OK
    })
}

/// What the title is told about a voice: `XSpeechSynthesizerVoiceInformation`.
#[repr(C)]
struct VoiceInformation {
    description: *const c_char,
    display_name: *const c_char,
    /// 0 female, 1 male.
    gender: u32,
    voice_id: *const c_char,
    language: *const c_char,
}

/// A host voice, as the title is told about it.
#[derive(Clone, Debug, PartialEq)]
struct Voice {
    id: String,
    display_name: String,
    description: String,
    language: String,
    male: bool,
}

/// Work for the speech thread, each with where its answer goes.
#[cfg(windows)]
enum Job {
    Speak {
        input: String,
        ssml: bool,
        voice: Option<String>,
        answer: std::sync::mpsc::SyncSender<Option<Vec<u8>>>,
    },
    Voices {
        answer: std::sync::mpsc::SyncSender<Vec<Voice>>,
    },
}

/// The speech thread's queue, started on first use.
#[cfg(windows)]
fn speech_thread() -> &'static std::sync::mpsc::Sender<Job> {
    static QUEUE: std::sync::OnceLock<std::sync::mpsc::Sender<Job>> = std::sync::OnceLock::new();
    QUEUE.get_or_init(|| {
        let (queue, jobs) = std::sync::mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("speech".into())
            .spawn(move || {
                use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx};
                // SAFETY: this thread's own apartment, entered once and
                // kept for its life.
                let entered = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
                if entered.is_err() {
                    tracing::warn!(?entered, "speech: no apartment on the speech thread");
                }
                let mut synthesizer = None;
                for job in jobs {
                    match job {
                        Job::Speak {
                            input,
                            ssml,
                            voice,
                            answer,
                        } => {
                            let wave = speak(&mut synthesizer, &input, ssml, voice.as_deref());
                            // The title may have stopped waiting.
                            let _ = answer.send(wave);
                        }
                        Job::Voices { answer } => {
                            let _ = answer.send(host_voices());
                        }
                    }
                }
            })
            .expect("the speech thread starts");
        queue
    })
}

/// Hands `job` to the speech thread and waits up to `PATIENCE` for what it
/// sends back on the other end of `made`; `None` if it does not answer.
#[cfg(windows)]
fn ask<T>(job: Job, made: std::sync::mpsc::Receiver<T>) -> Option<T> {
    if speech_thread().send(job).is_err() {
        return None;
    }
    let answer = made.recv_timeout(PATIENCE);
    if answer.is_err() {
        tracing::warn!("speech: the host did not answer in time; narration is silent");
    }
    answer.ok()
}

/// The host's installed voices; none where the host has no speech.
#[cfg(windows)]
fn voices() -> Vec<Voice> {
    let (answer, made) = std::sync::mpsc::sync_channel(1);
    ask(Job::Voices { answer }, made).unwrap_or_default()
}
#[cfg(not(windows))]
fn voices() -> Vec<Voice> {
    Vec::new()
}

/// The host's voices, on the speech thread.
#[cfg(windows)]
fn host_voices() -> Vec<Voice> {
    use windows::Media::SpeechSynthesis::{SpeechSynthesizer, VoiceGender};
    let list = || -> windows::core::Result<Vec<Voice>> {
        let mut voices = Vec::new();
        for voice in &SpeechSynthesizer::AllVoices()? {
            voices.push(Voice {
                id: voice.Id()?.to_string(),
                display_name: voice.DisplayName()?.to_string(),
                description: voice.Description()?.to_string(),
                language: voice.Language()?.to_string(),
                male: voice.Gender()? == VoiceGender::Male,
            });
        }
        Ok(voices)
    };
    list().unwrap_or_else(|error| {
        tracing::warn!(%error, "speech: the host's voices could not be listed");
        Vec::new()
    })
}

/// Speaks `input` (plain text, or SSML when `ssml`) in the host voice of id
/// `voice` (the default for `None`, or if that voice is gone) and returns
/// a 16 kHz mono 16-bit WAVE file, or `None` where the host has no working
/// voice or does not answer in time.
#[cfg(windows)]
fn synthesize(input: &str, ssml: bool, voice: Option<&str>) -> Option<Vec<u8>> {
    let (answer, made) = std::sync::mpsc::sync_channel(1);
    let job = Job::Speak {
        input: input.to_owned(),
        ssml,
        voice: voice.map(str::to_owned),
        answer,
    };
    ask(job, made).flatten()
}
#[cfg(not(windows))]
fn synthesize(_: &str, _: bool, _: Option<&str>) -> Option<Vec<u8>> {
    None
}

/// Speaks one line on the speech thread with its synthesizer, made on first
/// use and kept.
#[cfg(windows)]
fn speak(
    synthesizer: &mut Option<windows::Media::SpeechSynthesis::SpeechSynthesizer>,
    input: &str,
    ssml: bool,
    voice: Option<&str>,
) -> Option<Vec<u8>> {
    use windows::Media::SpeechSynthesis::SpeechSynthesizer;
    use windows::Storage::Streams::DataReader;
    use windows::core::HSTRING;

    let mut run = || -> windows::core::Result<Vec<u8>> {
        if synthesizer.is_none() {
            *synthesizer = Some(SpeechSynthesizer::new()?);
        }
        let synthesizer = synthesizer.as_ref().expect("made above");
        let picked = match voice {
            Some(id) => SpeechSynthesizer::AllVoices()?
                .into_iter()
                .find(|v| v.Id().is_ok_and(|v| v == id))
                .or_else(|| {
                    tracing::warn!(voice = id, "speech: voice gone; using the default");
                    None
                }),
            None => None,
        };
        synthesizer.SetVoice(&match picked {
            Some(picked) => picked,
            None => SpeechSynthesizer::DefaultVoice()?,
        })?;
        let input = HSTRING::from(input);
        let stream = if ssml {
            synthesizer.SynthesizeSsmlToStreamAsync(&input)?
        } else {
            synthesizer.SynthesizeTextToStreamAsync(&input)?
        }
        .join()?;
        // Narration is seconds of 16-bit audio: far below 4 GiB.
        let size = u32::try_from(stream.Size()?).unwrap_or(u32::MAX);
        let reader = DataReader::CreateDataReader(&stream.GetInputStreamAt(0)?)?;
        let loaded = reader.LoadAsync(size)?.join()?;
        let mut wave = vec![0; loaded as usize];
        reader.ReadBytes(&mut wave)?;
        Ok(wave)
    };
    match run() {
        Ok(wave) => {
            let converted = to_rate(&wave);
            if converted.is_none() {
                tracing::warn!(
                    "speech: the host's voice made no 16-bit PCM WAVE; narration is silent"
                );
            }
            converted
        }
        Err(error) => {
            tracing::warn!(%error, "speech: no host voice; narration is silent");
            // A broken synthesizer is made anew for the next line.
            *synthesizer = None;
            None
        }
    }
}

/// `wave` (16-bit PCM WAVE, any rate and channel count) as a WAVE file of
/// `RATE` Hz mono 16-bit PCM: channels averaged, resampled linearly. `None`
/// if `wave` is not 16-bit PCM WAVE.
fn to_rate(wave: &[u8]) -> Option<Vec<u8>> {
    let u16_at = |at: usize| Some(u16::from_le_bytes(wave.get(at..at + 2)?.try_into().ok()?));
    let u32_at = |at: usize| Some(u32::from_le_bytes(wave.get(at..at + 4)?.try_into().ok()?));
    if wave.get(..4)? != b"RIFF" || wave.get(8..12)? != b"WAVE" {
        return None;
    }
    let (mut format, mut data) = (None, None);
    let mut at = 12;
    while let (Some(id), Some(len)) = (wave.get(at..at + 4), u32_at(at + 4)) {
        let body = at + 8;
        let end = body.checked_add(len as usize)?;
        match id {
            // PCM (1) or WAVE_FORMAT_EXTENSIBLE (0xfffe) holding PCM.
            b"fmt " => {
                format = Some((
                    u16_at(body)?,
                    u16_at(body + 2)?,
                    u32_at(body + 4)?,
                    u16_at(body + 14)?,
                ))
            }
            b"data" => data = Some(wave.get(body..end.min(wave.len()))?),
            _ => {}
        }
        at = end + (len as usize & 1);
    }
    let (tag, channels, rate, bits) = format?;
    if !matches!(tag, 1 | 0xfffe) || bits != 16 || channels == 0 || rate == 0 {
        return None;
    }
    let frame = usize::from(channels) * 2;
    let mono: Vec<i32> = data?
        .chunks_exact(frame)
        .map(|f| {
            let sum: i32 = f
                .chunks_exact(2)
                .map(|s| i32::from(i16::from_le_bytes([s[0], s[1]])))
                .sum();
            sum / i32::from(channels)
        })
        .collect();
    let count = (mono.len() as u64 * u64::from(RATE) / u64::from(rate)) as usize;
    let mut out = Vec::with_capacity(44 + count * 2);
    out.extend_from_slice(&SILENCE);
    for i in 0..count {
        let x = i as u64 * u64::from(rate);
        let (j, part) = ((x / u64::from(RATE)) as usize, (x % u64::from(RATE)) as i64);
        let a = i64::from(mono[j]);
        let b = i64::from(*mono.get(j + 1).unwrap_or(&mono[j]));
        let s = a + (b - a) * part / i64::from(RATE);
        out.extend_from_slice(&(s as i16).to_le_bytes());
    }
    let samples = u32::try_from(count * 2).ok()?;
    out[4..8].copy_from_slice(&(36 + samples).to_le_bytes());
    out[40..44].copy_from_slice(&samples.to_le_bytes());
    Some(out)
}

type This = *const Object;

/// A stream of the title's words: the host's speech, or silence.
fn stream(synth: *const u8, words: *const c_char, ssml: bool, out: *mut *const u8) -> HRESULT {
    let Some(voice) = voice_of(synth) else {
        return super::xasync::E_INVALIDARG;
    };
    let words = text(words).unwrap_or_default();
    tracing::debug!(text = words, ssml, voice, "speech");
    let wave = synthesize(&words, ssml, voice.as_deref()).unwrap_or_else(|| SILENCE.to_vec());
    let handle = open(Entry::Stream(wave));
    let result = put(out, handle);
    if result != S_OK {
        close_handle(handle, true);
    }
    result
}

/// `XSpeechSynthesizerInstalledVoicesCallback`: true to go on.
type VoiceCallback = extern "system" fn(*const VoiceInformation, *mut c_void) -> bool;

/// Calls `callback` once for each host voice until it returns false. The
/// strings live until the call returns.
extern "system" fn enumerate_voices(
    _: This,
    context: *mut c_void,
    callback: Option<VoiceCallback>,
) -> HRESULT {
    let Some(callback) = callback else {
        return super::xasync::E_INVALIDARG;
    };
    let c = |s: &str| CString::new(s.replace('\0', "")).unwrap_or_default();
    for voice in voices() {
        let (description, display_name, id, language) = (
            c(&voice.description),
            c(&voice.display_name),
            c(&voice.id),
            c(&voice.language),
        );
        let information = VoiceInformation {
            description: description.as_ptr(),
            display_name: display_name.as_ptr(),
            gender: u32::from(voice.male),
            voice_id: id.as_ptr(),
            language: language.as_ptr(),
        };
        if !callback(&information, context) {
            break;
        }
    }
    S_OK
}
extern "system" fn create(_: This, out: *mut *const u8) -> HRESULT {
    let handle = open(Entry::Synthesizer(None));
    let result = put(out, handle);
    if result != S_OK {
        close_handle(handle, false);
    }
    result
}
extern "system" fn close(_: This, synth: *const u8) -> HRESULT {
    close_handle(synth, false)
}
extern "system" fn set_default_voice(_: This, synth: *const u8) -> HRESULT {
    if set_voice(synth, None) {
        S_OK
    } else {
        super::xasync::E_INVALIDARG
    }
}
/// Picks the host voice whose id the title read from `enumerate_voices`.
extern "system" fn set_custom_voice(_: This, synth: *const u8, voice: *const c_char) -> HRESULT {
    let Some(voice) = text(voice) else {
        return super::xasync::E_INVALIDARG;
    };
    if voice_of(synth).is_none() {
        return super::xasync::E_INVALIDARG;
    }
    if !voices().iter().any(|v| v.id == voice) {
        tracing::warn!(
            voice,
            "speech: the title picked a voice the host does not have"
        );
        return super::xasync::E_INVALIDARG;
    }
    if set_voice(synth, Some(voice)) {
        S_OK
    } else {
        super::xasync::E_INVALIDARG
    }
}
extern "system" fn create_stream(
    _: This,
    synth: *const u8,
    words: *const c_char,
    out: *mut *const u8,
) -> HRESULT {
    stream(synth, words, false, out)
}
extern "system" fn create_stream_ssml(
    _: This,
    synth: *const u8,
    ssml: *const c_char,
    out: *mut *const u8,
) -> HRESULT {
    stream(synth, ssml, true, out)
}
extern "system" fn close_stream(_: This, stream: *const u8) -> HRESULT {
    close_handle(stream, true)
}
extern "system" fn stream_size(_: This, stream: *const u8, out: *mut usize) -> HRESULT {
    match stream_len(stream) {
        Some(len) => put(out, len),
        None => super::xasync::E_INVALIDARG,
    }
}
extern "system" fn stream_data(
    _: This,
    stream: *const u8,
    size: usize,
    buffer: *mut u8,
    used: *mut usize,
) -> HRESULT {
    handles(|table| {
        let Some(Entry::Stream(wave)) = table.get(&(stream as usize)) else {
            return super::xasync::E_INVALIDARG;
        };
        if buffer.is_null() || size < wave.len() {
            return super::xasync::E_NOT_SUFFICIENT_BUFFER;
        }
        // SAFETY: the buffer holds `size` bytes, checked above.
        unsafe { buffer.copy_from_nonoverlapping(wave.as_ptr(), wave.len()) };
        if !used.is_null() {
            // SAFETY: checked non-null.
            unsafe { used.write(wave.len()) };
        }
        S_OK
    })
}

static VTABLE: Methods<22> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    missing::<3> as *const (),       // 0x18 closed captions
    missing::<4> as *const (),       // 0x20
    missing::<5> as *const (),       // 0x28 high contrast
    missing::<6> as *const (),       // 0x30 speech to text
    missing::<7> as *const (),       // 0x38
    enumerate_voices as *const (),   // 0x40
    create as *const (),             // 0x48
    close as *const (),              // 0x50
    set_default_voice as *const (),  // 0x58
    set_custom_voice as *const (),   // 0x60
    create_stream as *const (),      // 0x68 from text
    close_stream as *const (),       // 0x70 a stream
    stream_size as *const (),        // 0x78
    stream_data as *const (),        // 0x80
    missing::<17> as *const (),      // 0x88 speech to text
    missing::<18> as *const (),      // 0x90
    missing::<19> as *const (),      // 0x98
    missing::<20> as *const (),      // 0xa0
    create_stream_ssml as *const (), // 0xa8 from SSML
]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::xgameruntime::method;

    type Create = extern "system" fn(This, *mut *const u8) -> HRESULT;
    type Stream = extern "system" fn(This, *const u8, *const c_char, *mut *const u8) -> HRESULT;
    type Size = extern "system" fn(This, *const u8, *mut usize) -> HRESULT;
    type Data = extern "system" fn(This, *const u8, usize, *mut u8, *mut usize) -> HRESULT;
    type Close = extern "system" fn(This, *const u8) -> HRESULT;
    type Enumerate = extern "system" fn(This, *mut c_void, Option<VoiceCallback>) -> HRESULT;
    type SetVoice = extern "system" fn(This, *const u8, *const c_char) -> HRESULT;

    const SSML: &std::ffi::CStr =
        c"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>Hello</speak>";

    /// The WAVE file of `words`, through the slots the title calls.
    fn spoken(slot: usize, words: &std::ffi::CStr) -> Vec<u8> {
        let this: This = &OBJECT;
        let (mut synth, mut stream, mut size) = (std::ptr::null(), std::ptr::null(), 0);
        assert_eq!(method::<Create>(&OBJECT, 0x48)(this, &mut synth), S_OK);
        assert_eq!(
            method::<Stream>(&OBJECT, slot)(this, synth, words.as_ptr(), &mut stream),
            S_OK
        );
        assert_eq!(method::<Size>(&OBJECT, 0x78)(this, stream, &mut size), S_OK);
        let mut wave = vec![0u8; size];
        let mut used = 0;
        let data = method::<Data>(&OBJECT, 0x80);
        assert_eq!(
            data(this, stream, size - 1, wave.as_mut_ptr(), &mut used),
            super::super::xasync::E_NOT_SUFFICIENT_BUFFER,
            "a buffer one byte short"
        );
        assert_eq!(data(this, stream, size, wave.as_mut_ptr(), &mut used), S_OK);
        assert_eq!(used, size);
        assert_eq!(method::<Close>(&OBJECT, 0x70)(this, stream), S_OK);
        assert_eq!(method::<Close>(&OBJECT, 0x50)(this, synth), S_OK);
        wave
    }

    /// The format chunk of a WAVE file: (tag, channels, rate, bits).
    fn format(wave: &[u8]) -> (u16, u16, u32, u16) {
        let u16_at = |at: usize| u16::from_le_bytes(wave[at..at + 2].try_into().unwrap());
        let rate = u32::from_le_bytes(wave[24..28].try_into().unwrap());
        (u16_at(20), u16_at(22), rate, u16_at(34))
    }

    #[test]
    fn streams_are_16_khz_mono_wave_files() {
        for (slot, words) in [(0x68, c"Hello"), (0xa8, SSML)] {
            let wave = spoken(slot, words);
            assert_eq!(
                (&wave[..4], &wave[8..16], &wave[36..40]),
                (&b"RIFF"[..], &b"WAVEfmt "[..], &b"data"[..])
            );
            assert_eq!(format(&wave), (1, 1, RATE, 16));
            assert_eq!(
                u32::from_le_bytes(wave[40..44].try_into().unwrap()) as usize,
                wave.len() - 44
            );
        }
    }

    #[test]
    fn silence_is_an_empty_wave_file() {
        assert_eq!(format(&SILENCE), (1, 1, RATE, 16));
        assert_eq!(u32::from_le_bytes(SILENCE[40..44].try_into().unwrap()), 0);
    }

    /// Text and SSML both come back as audio where the host has a voice.
    #[test]
    fn the_host_voice_speaks_text_and_ssml() {
        let Some(text) = synthesize("Hello", false, None) else {
            eprintln!("SKIPPED the_host_voice_speaks_text_and_ssml: no host voice");
            return;
        };
        let ssml =
            synthesize(SSML.to_str().unwrap(), true, None).expect("SSML spoken where text is");
        for wave in [text, ssml] {
            // "Hello" lasts well over a tenth of a second.
            assert!(
                wave.len() > 44 + RATE as usize / 5,
                "{} bytes is not speech",
                wave.len()
            );
            assert_eq!(format(&wave), (1, 1, RATE, 16));
        }
    }

    /// The crash a reviewer found: lines spoken on several title threads at
    /// once ended the process with 0xc0000409.
    #[test]
    fn lines_spoken_on_many_threads_at_once_all_come_back() {
        let threads: Vec<_> = (0..8)
            .map(|i| {
                std::thread::spawn(move || {
                    spoken(
                        if i % 2 == 0 { 0x68 } else { 0xa8 },
                        if i % 2 == 0 { c"Hello" } else { SSML },
                    )
                })
            })
            .collect();
        for thread in threads {
            let wave = thread.join().expect("no thread panicked");
            assert_eq!(format(&wave), (1, 1, RATE, 16));
        }
    }

    #[test]
    fn stale_and_foreign_handles_are_refused() {
        let this: This = &OBJECT;
        let invalid = super::super::xasync::E_INVALIDARG;
        let mut synth = std::ptr::null();
        assert_eq!(method::<Create>(&OBJECT, 0x48)(this, &mut synth), S_OK);
        let mut stream = std::ptr::null();
        assert_eq!(
            method::<Stream>(&OBJECT, 0x68)(this, synth, c"Hi".as_ptr(), &mut stream),
            S_OK
        );
        let close_stream = method::<Close>(&OBJECT, 0x70);
        let close_synth = method::<Close>(&OBJECT, 0x50);
        assert_eq!(
            close_synth(this, stream),
            invalid,
            "a stream is not a synthesizer"
        );
        assert_eq!(
            close_stream(this, synth),
            invalid,
            "a synthesizer is not a stream"
        );
        assert_eq!(close_stream(this, stream), S_OK);
        assert_eq!(close_stream(this, stream), invalid, "closed twice");
        let mut size = 0;
        assert_eq!(
            method::<Size>(&OBJECT, 0x78)(this, stream, &mut size),
            invalid,
            "read after close"
        );
        let bogus = 0x1234usize as *const u8;
        assert_eq!(
            method::<Size>(&OBJECT, 0x78)(this, bogus, &mut size),
            invalid
        );
        assert_eq!(
            method::<Stream>(&OBJECT, 0x68)(this, bogus, c"Hi".as_ptr(), &mut stream),
            invalid
        );
        assert_eq!(close_synth(this, synth), S_OK);
        assert_eq!(close_synth(this, synth), invalid, "closed twice");
    }

    #[test]
    fn resampling_averages_channels_and_keeps_the_duration() {
        // One second of 22,050 Hz stereo: left +1000, right -200.
        let (rate, channels, frames) = (22_050u32, 2u16, 22_050usize);
        let mut wave = Vec::new();
        wave.extend_from_slice(b"RIFF");
        wave.extend_from_slice(&(36 + frames as u32 * 4).to_le_bytes());
        wave.extend_from_slice(b"WAVEfmt \x10\0\0\0\x01\0");
        wave.extend_from_slice(&channels.to_le_bytes());
        wave.extend_from_slice(&rate.to_le_bytes());
        wave.extend_from_slice(&(rate * 4).to_le_bytes());
        wave.extend_from_slice(b"\x04\0\x10\0data");
        wave.extend_from_slice(&(frames as u32 * 4).to_le_bytes());
        for _ in 0..frames {
            wave.extend_from_slice(&1000i16.to_le_bytes());
            wave.extend_from_slice(&(-200i16).to_le_bytes());
        }
        let out = to_rate(&wave).unwrap();
        assert_eq!(format(&out), (1, 1, RATE, 16));
        assert_eq!(out.len(), 44 + RATE as usize * 2, "one second at 16 kHz");
        assert!(
            out[44..]
                .chunks_exact(2)
                .all(|s| i16::from_le_bytes([s[0], s[1]]) == 400)
        );
        assert_eq!(to_rate(b"RIFF\0\0\0\0WAVE"), None, "no format chunk");
        assert_eq!(to_rate(b"not a wave file"), None);
    }

    /// The voices the title is told about, read as the title reads them.
    fn listed() -> Vec<Voice> {
        extern "system" fn collect(
            information: *const VoiceInformation,
            context: *mut c_void,
        ) -> bool {
            // SAFETY: the record and its strings live for this call.
            let (info, list) = unsafe { (&*information, &mut *context.cast::<Vec<Voice>>()) };
            let s = |p: *const c_char| text(p).expect("a string, not null");
            list.push(Voice {
                id: s(info.voice_id),
                display_name: s(info.display_name),
                description: s(info.description),
                language: s(info.language),
                male: match info.gender {
                    0 => false,
                    1 => true,
                    g => panic!("gender {g} is neither female (0) nor male (1)"),
                },
            });
            true
        }
        let mut list = Vec::<Voice>::new();
        let context = (&raw mut list).cast::<c_void>();
        assert_eq!(
            method::<Enumerate>(&OBJECT, 0x40)(&OBJECT, context, Some(collect)),
            S_OK
        );
        list
    }

    #[test]
    fn the_title_is_told_the_hosts_voices() {
        let listed = listed();
        assert_eq!(listed, voices());
        for voice in &listed {
            assert!(
                !voice.id.is_empty() && !voice.display_name.is_empty(),
                "{voice:?}"
            );
        }
        // A callback that returns false stops the listing after one voice.
        extern "system" fn first(_: *const VoiceInformation, context: *mut c_void) -> bool {
            // SAFETY: the test's counter.
            unsafe { *context.cast::<usize>() += 1 };
            false
        }
        let mut count = 0usize;
        let context = (&raw mut count).cast::<c_void>();
        assert_eq!(
            method::<Enumerate>(&OBJECT, 0x40)(&OBJECT, context, Some(first)),
            S_OK
        );
        assert_eq!(count, listed.len().min(1));
        assert_eq!(
            method::<Enumerate>(&OBJECT, 0x40)(&OBJECT, std::ptr::null_mut(), None),
            super::super::xasync::E_INVALIDARG
        );
    }

    #[test]
    fn a_picked_voice_speaks_and_an_unknown_one_is_refused() {
        let this: This = &OBJECT;
        let mut synth = std::ptr::null();
        assert_eq!(method::<Create>(&OBJECT, 0x48)(this, &mut synth), S_OK);
        let set = method::<SetVoice>(&OBJECT, 0x60);
        assert_eq!(
            set(this, synth, c"no such voice".as_ptr()),
            super::super::xasync::E_INVALIDARG
        );
        assert_eq!(
            set(this, std::ptr::null(), c"x".as_ptr()),
            super::super::xasync::E_INVALIDARG
        );
        if let Some(last) = listed().last() {
            let id = CString::new(last.id.clone()).unwrap();
            assert_eq!(set(this, synth, id.as_ptr()), S_OK);
            assert_eq!(voice_of(synth), Some(Some(last.id.clone())));
            let mut stream = std::ptr::null();
            let say = method::<Stream>(&OBJECT, 0x68);
            assert_eq!(say(this, synth, c"Hello".as_ptr(), &mut stream), S_OK);
            let spoken = stream_len(stream).unwrap();
            assert!(
                spoken > 44 + RATE as usize / 5,
                "{spoken} bytes in {}'s voice",
                last.display_name
            );
            assert_eq!(method::<Close>(&OBJECT, 0x70)(this, stream), S_OK);
        } else {
            eprintln!("SKIPPED a_picked_voice_speaks_and_an_unknown_one_is_refused: no host voice");
        }
        assert_eq!(
            method::<Close>(&OBJECT, 0x58)(this, synth),
            S_OK,
            "back to the default voice"
        );
        assert_eq!(voice_of(synth), Some(None));
        assert_eq!(method::<Close>(&OBJECT, 0x50)(this, synth), S_OK);
    }
}
