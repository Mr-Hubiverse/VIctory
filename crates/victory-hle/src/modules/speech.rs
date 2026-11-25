//! Accessibility: speech synthesis for the title's narration. Text and SSML
//! are spoken by the host's default voice (Windows speech synthesis) into a
//! WAVE file the title reads back; where the host has no working voice, the
//! stream is valid WAVE audio of no length and narration is silent.
// shortcut: no voices are listed and a custom voice is ignored, so the
// title always gets the host's default voice. Listing needs the layout of
// the title's voice information records (slot 0x40).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, c_void};

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

/// A synthesizer holds nothing (each stream asks the host anew), so its
/// handle is a fixed address and closing it is free.
static SYNTHESIZER: u8 = 0;

type This = *const Object;

/// Speaks `input` (plain text, or SSML when `ssml`) with the host's default
/// voice and returns the WAVE file the host made, or `None` where the host
/// has no working voice.
#[cfg(windows)]
fn synthesize(input: &str, ssml: bool) -> Option<Vec<u8>> {
    use windows::Media::SpeechSynthesis::SpeechSynthesizer;
    use windows::Storage::Streams::DataReader;
    use windows::core::HSTRING;

    let speak = || -> windows::core::Result<Vec<u8>> {
        let synthesizer = SpeechSynthesizer::new()?;
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
fn synthesize(_: &str, _: bool) -> Option<Vec<u8>> {
    None
}

/// A stream of the title's words: the host's speech, or silence.
fn stream(words: *const c_char, ssml: bool, out: *mut *const u8) -> HRESULT {
    let words = text(words).unwrap_or_default();
    tracing::debug!(text = words, ssml, "speech");
    let wave = synthesize(&words, ssml).unwrap_or_else(|| SILENCE.to_vec());
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

/// No voice is listed, so the callback never runs.
extern "system" fn enumerate_voices(
    _: This,
    _context: *mut c_void,
    _callback: *const c_void,
) -> HRESULT {
    S_OK
}
extern "system" fn create(_: This, out: *mut *const u8) -> HRESULT {
    put(out, &raw const SYNTHESIZER)
}
extern "system" fn close(_: This, _: *const u8) -> HRESULT {
    S_OK
}
extern "system" fn set_default_voice(_: This, _: *const u8) -> HRESULT {
    S_OK
}
extern "system" fn set_custom_voice(_: This, _: *const u8, _voice: *const c_char) -> HRESULT {
    S_OK
}
extern "system" fn create_stream(
    _: This,
    _: *const u8,
    words: *const c_char,
    out: *mut *const u8,
) -> HRESULT {
    stream(words, false, out)
}
extern "system" fn create_stream_ssml(
    _: This,
    _: *const u8,
    ssml: *const c_char,
    out: *mut *const u8,
) -> HRESULT {
    stream(ssml, true, out)
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
        let Some(text) = synthesize("Hello", false) else {
            eprintln!("SKIPPED the_host_voice_speaks_text_and_ssml: no host voice");
            return;
        };
        let ssml = synthesize(
            "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>Hello</speak>",
            true,
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
}
