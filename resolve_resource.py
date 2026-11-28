import resolve1

P = 'crates/victory-gpu/src/resource.rs'
src, hs = resolve1.hunks(P)

h1 = """            Some(texture) => Ok((texture, None, 0, false)),
            None => host_texture(&r.desc.from_level(base), r.host.state, clear),
"""

# h10: ours' host_texture_as body + theirs' residency grafts
h10 = hs[10][1]
h10 = h10.replace(
    """    if let Err(e) = made {
        let why = no_texture(e.code().0);""",
    """    if let Err(e) = made {
        crate::residency::allocation_error(&e);
        let why = no_texture(e.code().0);""")
h10 = h10.replace(
    """    let texture = texture.ok_or(NoTexture::Refused)?;
    crate::host::name(&texture, || {
        format!("texture {}x{} f{} flags {:#x}", d.width, d.height, d.format, d.flags)
    });
    Ok((texture, None, skip, false))""",
    """    let texture = texture.ok_or(NoTexture::Refused)?;
    if !reserved
        && let Err(e) = crate::host::resident(&host.device, &texture)
            .and_then(|()| crate::host::mapped(&host.device))
    {
        tracing::warn!(error = %e, desc = ?d, "a host texture is not resident");
        return Err(NoTexture::ForNow);
    }
    crate::host::name(&texture, || {
        format!("texture {}x{} f{} flags {:#x}", d.width, d.height, d.format, d.flags)
    });
    Ok((texture, None, skip, false))""")
assert 'resident(&host.device' in h10 and 'allocation_error' in h10

choices = ['both', h1, 'both', 'theirs', 'theirs', 'both', 'both', 'both',
           'both', 'both', h10, 'both', 'ours', 'ours', 'both', 'both']
resolve1.resolve(P, choices)
