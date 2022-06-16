import pytest  # noqa: F401
from nle import nethack

from hackrl.models import baseline


class TestIdPairs:
    def test_id_pairs_func(self):
        table = baseline.id_pairs_table()

        for glyph, (id_, _type) in enumerate(table):
            assert baseline.id_pairs_func(glyph) == id_

    def test_id_pairs_table(self):
        table = baseline.id_pairs_table()
        for glyph, (id_, _type) in enumerate(table):
            if nethack.glyph_is_monster(glyph):
                assert nethack.glyph_to_mon(glyph) == nethack.glyph_to_mon(id_)
            elif nethack.glyph_is_body(glyph):
                assert nethack.glyph_to_mon(
                    glyph - nethack.GLYPH_BODY_OFF
                ) == nethack.glyph_to_mon(id_)
            elif nethack.glyph_is_object(glyph) and not nethack.glyph_is_statue(glyph):
                # ids are mons (NUMMONS), invisible (1), objects (NUM_OBJECTS) ...
                assert nethack.glyph_to_obj(glyph) == nethack.glyph_to_obj(
                    nethack.GLYPH_OBJ_OFF + id_ - nethack.NUMMONS - 1
                )

            # And so on.

        # nethack.NUM_OBJECTS replaced by baseline.NUM_OBJECTS below

        # Number of distinct ids in pairs.
        num_ids = len(set(x for (x, y) in table))

        # Number of monsters.
        assert nethack.GLYPH_PET_OFF == nethack.NUMMONS

        # Number of objects.
        assert nethack.GLYPH_CMAP_OFF - nethack.GLYPH_OBJ_OFF == baseline.NUM_OBJECTS

        # Number of cmap glyphs. This is NOT MAXPCHARS contrary to what
        # display.h claims!
        assert (
            nethack.GLYPH_EXPLODE_OFF - nethack.GLYPH_CMAP_OFF
            == nethack.MAXPCHARS - baseline.MAXEXPCHARS
        )
        num_cmap = nethack.MAXPCHARS - baseline.MAXEXPCHARS

        # Check that this is "most".
        assert abs(num_ids - (nethack.NUMMONS + baseline.NUM_OBJECTS + num_cmap)) < 30

    def test_id_pairs_table_groups(self):
        def glyph_is_explode(glyph):
            return nethack.GLYPH_EXPLODE_OFF <= glyph < nethack.GLYPH_ZAP_OFF

        def glyph_is_zap(glyph):
            return nethack.GLYPH_ZAP_OFF <= glyph < nethack.GLYPH_SWALLOW_OFF

        type_to_test = {
            baseline.GlyphGroup.MON: nethack.glyph_is_normal_monster,
            baseline.GlyphGroup.PET: nethack.glyph_is_pet,
            baseline.GlyphGroup.INVIS: nethack.glyph_is_invisible,
            baseline.GlyphGroup.DETECT: nethack.glyph_is_detected_monster,
            baseline.GlyphGroup.BODY: nethack.glyph_is_body,
            baseline.GlyphGroup.RIDDEN: nethack.glyph_is_ridden_monster,
            baseline.GlyphGroup.OBJ: nethack.glyph_is_object,
            baseline.GlyphGroup.CMAP: nethack.glyph_is_cmap,
            baseline.GlyphGroup.EXPLODE: glyph_is_explode,
            baseline.GlyphGroup.ZAP: glyph_is_zap,
            baseline.GlyphGroup.SWALLOW: nethack.glyph_is_swallow,
            baseline.GlyphGroup.WARNING: nethack.glyph_is_warning,
            baseline.GlyphGroup.STATUE: nethack.glyph_is_statue,
        }

        for glyph, (_id, type_) in enumerate(baseline.id_pairs_table()):
            assert type_to_test[type_](glyph)
