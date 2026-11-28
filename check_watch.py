import re, sys
src = open('crates/victory-gpu/src/watch.rs', encoding='utf-8').read()
names = ['stretch','title_memory_view','view_guard','shown','aliases','forget','pinned','layout',
 'left_alone','rests','opened_at','alias','unalias','unguard','open_here','cover','light_turn',
 'light_counts','known','since','page_writes','written_since_taken','landed_lately','landed_beside',
 'put_beside','views_mapped','test_books','stamp','track','clean','asks','guards_any',
 'block_published','remapped','title_memory_remapped','title_memory_reading','title_memory_reading_now',
 'title_memory_reading_range','title_memory_written','title_memory_remapping_range','title_memory_unmapping',
 'title_memory_changed','title_memory_freed','allocation_end','title_memory_arrived','title_memory_made_real',
 'use_lazy_views','use_soft_views','soft_view_guards','soft_runs','soft_block','once','writing','wrote',
 'materialize','soft','try_open_guarded','light','published','publish','nothing_new','unchanged_known',
 'writes_guarded','region_epoch','subpages','dirty4k','dirty4k_kept','dirty4k_count','parts_written',
 'landing','changes','again_after_change','lost','asked','writes','ask_ahead','ask_ahead_done','ask_once',
 'ask_once_end','known_view','wrote_own','guarded','open','settle','near','journal','journal_mut',
 'kernel_written','guard','hold_all','next_asking','note_hot','told','light_kept','light_mode','light_asks',
 'failed_asking','slow','view_known','opening','settle','protect_alias','readable_view','change_over',
 'change_done','came_back','drop','new','piece','set','swap_on','word','take','clear','clear_all','any','all']
from collections import Counter
defs = re.findall(r'(?m)^\s*(?:pub\(crate\)\s+|pub\s+)?(?:unsafe\s+)?fn (\w+)', src)
cnt = Counter(defs)
for n in names:
    c = cnt.get(n, 0)
    if c != 1:
        print(f'{n}: {c}')
