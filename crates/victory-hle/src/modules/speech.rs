//! Accessibility: speech synthesis for the title's narration. Text and SSML
//! are spoken by a host voice (Windows speech synthesis) into a WAVE file
//! the title reads back; where the host has no working voice, the stream is
//! valid WAVE audio of no length and narration is silent. The host's voices
//! are listed to the title, which may pick one for a synthesizer.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CString, c_char, c_void};
use std::sync::Mutex;

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

/// A synthesizer: the voice the title picked, or none for the host's
/// default. Each stream asks the host anew.
#[derive(Default)]
struct Synthesizer {
    voice: Mutex<Option<String>>,
}

/// The synthesizer behind a handle from `create`.
fn synthesizer<'a>(handle: *const u8) -> Option<&'a Synthesizer> {
    // SAFETY: the title passes a handle `create` made and has not closed.
    (!handle.is_null()).then(|| unsafe { &*handle.cast::<Synthesizer>() })
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
#[derive(Debug, PartialEq)]
struct Voice {
    id: String,
    display_name: String,
    description: String,
    language: String,
    male: bool,
}

/// The host's installed voices; none where the host has no speech.
#[cfg(windows)]
fn voices() -> Vec<Voice> {
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
#[cfg(not(windows))]
fn voices() -> Vec<Voice> {
    Vec::new()
}

type This = *const Object;

/// Speaks `input` (plain text, or SSML when `ssml`) with the host voice of
/// id `voice` (the default voice for `None`, or if that voice is gone) and
/// returns the WAVE file the host made, or `None` where the host has no
/// working voice.
#[cfg(windows)]
fn synthesize(input: &str, ssml: bool, voice: Option<&str>) -> Option<Vec<u8>> {
    use windows::Media::SpeechSynthesis::SpeechSynthesizer;
    use windows::Storage::Streams::DataReader;
    use windows::core::HSTRING;

    let speak = || -> windows::core::Result<Vec<u8>> {
        let synthesizer = SpeechSynthesizer::new()?;
        if let Some(id) = voice {
            let picked = SpeechSynthesizer::AllVoices()?
                .into_iter()
                .find(|v| v.Id().is_ok_and(|v| v == id));
            match picked {
                Some(picked) => synthesizer.SetVoice(&picked)?,
                None => tracing::warn!(voice = id, "speech: voice gone; using the default"),
            }
        }
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
    match speak() {
        Ok(wave) if wave.starts_with(b"RIFF") => Some(wave),
        Ok(_) => {
            tracing::warn!("speech: the host's voice made no WAVE file; narration is silent");
            None
        }
        Err(error) => {
            tracing::warn!(%error, "speech: no host voice; narration is silent");
            None
        }
    }
}
#[cfg(not(windows))]
fn synthesize(_: &str, _: bool, _: Option<&str>) -> Option<Vec<u8>> {
    None
}

/// A stream of the title's words: the host's speech, or silence.
fn stream(synth: *const u8, words: *const c_char, ssml: bool, out: *mut *const u8) -> HRESULT {
    let Some(synth) = synthesizer(synth) else {
        return super::xasync::E_INVALIDARG;
    };
    let voice = synth
        .voice
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let words = text(words).unwrap_or_default();
    tracing::debug!(text = words, ssml, voice, "speech");
    let wave = synthesize(&words, ssml, voice.as_deref()).unwrap_or_else(|| SILENCE.to_vec());
    let handle = Box::into_raw(Box::new(wave)).cast_const().cast::<u8>();
    let result = put(out, handle);
    if result != S_OK {
        // SAFETY: made just above and not handed out.
        drop(unsafe { Box::from_raw(handle.cast_mut().cast::<Vec<u8>>()) });
    }
    result
}

/// The WAVE file behind a stream handle from `stream`.
fn wave<'a>(stream: *const u8) -> Option<&'a [u8]> {
    // SAFETY: the title passes a handle `stream` made and has not closed.
    (!stream.is_null()).then(|| unsafe { (*stream.cast::<Vec<u8>>()).as_slice() })
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
    let handle = Box::into_raw(Box::<Synthesizer>::default())
        .cast_const()
        .cast::<u8>();
    let result = put(out, handle);
    if result != S_OK {
        // SAFETY: made just above and not handed out.
        drop(unsafe { Box::from_raw(handle.cast_mut().cast::<Synthesizer>()) });
    }
    result
}
extern "system" fn close(_: This, synth: *const u8) -> HRESULT {
    if !synth.is_null() {
        // SAFETY: a handle `create` made, closed once by the title.
        drop(unsafe { Box::from_raw(synth.cast_mut().cast::<Synthesizer>()) });
    }
    S_OK
}
extern "system" fn set_default_voice(_: This, synth: *const u8) -> HRESULT {
    let Some(synth) = synthesizer(synth) else {
        return super::xasync::E_INVALIDARG;
    };
    *synth.voice.lock().unwrap_or_else(|e| e.into_inner()) = None;
    S_OK
}
/// Picks the host voice whose id the title read from `enumerate_voices`.
extern "system" fn set_custom_voice(_: This, synth: *const u8, voice: *const c_char) -> HRESULT {
    let (Some(synth), Some(voice)) = (synthesizer(synth), text(voice)) else {
        return super::xasync::E_INVALIDARG;
    };
    if !voices().iter().any(|v| v.id == voice) {
        tracing::warn!(
            voice,
            "speech: the title picked a voice the host does not have"
        );
        return super::xasync::E_INVALIDARG;
    }
    *synth.voice.lock().unwrap_or_else(|e| e.into_inner()) = Some(voice);
    S_OK
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
    if !stream.is_null() {
        // SAFETY: a handle `stream` made, closed once by the title.
        drop(unsafe { Box::from_raw(stream.cast_mut().cast::<Vec<u8>>()) });
    }
    S_OK
}
extern "system" fn stream_size(_: This, stream: *const u8, out: *mut usize) -> HRESULT {
    match wave(stream) {
        Some(wave) => put(out, wave.len()),
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
    let Some(wave) = wave(stream) else {
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

    #[test]
    fn streams_are_wave_files() {
        for (slot, words) in [
            (0x68, c"Hello"),
            (0xa8, c"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>Hello</speak>"),
        ] {
            let wave = spoken(slot, words);
            assert_eq!((&wave[..4], &wave[8..16]), (&b"RIFF"[..], &b"WAVEfmt "[..]));
        }
    }

    #[test]
    fn silence_is_an_empty_wave_file() {
        assert_eq!(
            (&SILENCE[..4], &SILENCE[8..16], &SILENCE[36..40]),
            (&b"RIFF"[..], &b"WAVEfmt "[..], &b"data"[..])
        );
        assert_eq!(
            u32::from_le_bytes(SILENCE[24..28].try_into().unwrap()),
            16_000
        );
        assert_eq!(u32::from_le_bytes(SILENCE[40..44].try_into().unwrap()), 0);
    }

    /// Text and SSML both come back as audio where the host has a voice.
    #[test]
    fn the_host_voice_speaks_text_and_ssml() {
        let Some(text) = synthesize("Hello", false, None) else {
            eprintln!("SKIPPED the_host_voice_speaks_text_and_ssml: no host voice");
            return;
        };
        let ssml = synthesize(
            "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>Hello</speak>",
            true,
            None,
        )
        .expect("SSML spoken where text is");
        for wave in [text, ssml] {
            assert!(
                wave.len() > SILENCE.len() + 1000,
                "{} bytes is not speech",
                wave.len()
            );
            assert_eq!(&wave[8..12], b"WAVE");
        }
    }

    type Enumerate = extern "system" fn(This, *mut c_void, Option<VoiceCallback>) -> HRESULT;
    type SetVoice = extern "system" fn(This, *const u8, *const c_char) -> HRESULT;

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
        let listed = listed();
        if let Some(last) = listed.last() {
            let id = CString::new(last.id.clone()).unwrap();
            assert_eq!(set(this, synth, id.as_ptr()), S_OK);
            assert_eq!(
                synthesizer(synth).unwrap().voice.lock().unwrap().as_deref(),
                Some(last.id.as_str())
            );
            let mut stream = std::ptr::null();
            let say = method::<Stream>(&OBJECT, 0x68);
            assert_eq!(say(this, synth, c"Hello".as_ptr(), &mut stream), S_OK);
            let spoken = wave(stream).unwrap().len();
            assert!(
                spoken > SILENCE.len() + 1000,
                "{spoken} bytes in {}'s voice",
                last.display_name
            );
            assert_eq!(method::<Close>(&OBJECT, 0x70)(this, stream), S_OK);
        } else {
            eprintln!("SKIPPED a_picked_voice_speaks...: no host voice");
        }
        assert_eq!(
            method::<Close>(&OBJECT, 0x58)(this, synth),
            S_OK,
            "back to the default voice"
        );
        assert_eq!(
            synthesizer(synth).unwrap().voice.lock().unwrap().as_deref(),
            None
        );
        assert_eq!(method::<Close>(&OBJECT, 0x50)(this, synth), S_OK);
    }
}
