    use super::{CARD_COPY, HOST_COPY, Landings, READ_BACK};

    /// A frame's line comes when the next frame's first landing does: read
    /// backs summed per destination with their texture and the earliest
    /// frame they were recorded in, a card copy merged back, and a host
    /// copy found unchanged only counted.
    #[test]
    fn a_frames_landings_are_one_line_once_the_next_frame_begins() {
        let mut l = Landings::default();
        assert!(l.add(5, READ_BACK, 0x1000, 0x200, 0x100, 0x100, 0xabc, Some(4)).is_none());
        assert!(l.add(5, READ_BACK, 0x1000, 0x200, 0x100, 0x100, 0xabc, Some(3)).is_none());
        assert!(l.add(5, CARD_COPY, 0x2000, 0x40, 0x40, 0x10, 0, None).is_none());
        assert!(l.add(5, HOST_COPY, 0x3000, 0x40, 0x40, 0, 0, None).is_none());
        let (frame, count, unchanged, landed) = l.add(6, CARD_COPY, 0x2000, 0x40, 0x40, 0x40, 0, None).expect("frame 5's line");
        assert_eq!((frame, count, unchanged), (5, 2, 1));
        assert_eq!(
            landed,
            [
                "card copy 0x2000+0x40: x1, 64 bytes, 16 written",
                "read-back 0x1000+0x200: x2, 512 bytes, 512 written, from 0xabc, recorded in frame 3",
            ]
        );
        // Frame 6 holds its own.
        assert_eq!(l.take().map(|t| (t.0, t.1)), Some((6, 1)));
        assert!(l.take().is_none());
    }
