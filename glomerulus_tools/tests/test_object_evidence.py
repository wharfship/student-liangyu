import unittest
from PIL import Image
from glomerulus_tools.object_evidence import crop_square,prepare_gncnn_input


class EvidenceCoordinateTests(unittest.TestCase):
    def test_boundary_padding_preserves_source_coordinates(self):
        source=Image.new('RGB',(100,80),(20,30,40))
        crop,box=crop_square(source,[0,0,20,20],60)
        self.assertEqual(box,[-20,-20,40,40])
        self.assertEqual(crop.getpixel((19,19)),(255,255,255))
        self.assertEqual(crop.getpixel((20,20)),(20,30,40))
        self.assertEqual(source.size,(100,80))

    def test_rectangular_candidate_uses_center_without_stretch(self):
        source=Image.new('RGB',(200,100),(20,30,40))
        crop,box=crop_square(source,[50,20,150,40],80)
        self.assertEqual(box,[60,-10,140,70])
        self.assertEqual(crop.size,(80,80))

    def test_gncnn_matches_author_resize_without_intermediate_downsample(self):
        # Small structures disappear differently through512; compare real pixel results.
        source=Image.new('RGB',(4096,4096),'white')
        for x in range(1200,1300):
            for y in range(1200,1205):source.putpixel((x,y),(90,10,200))
        actual=prepare_gncnn_input(source,dict(input_recipe='author_qupath4096_resize800'))
        expected=source.resize((800,800),Image.Resampling.BILINEAR)
        self.assertEqual(actual.tobytes(),expected.tobytes())
        self.assertNotEqual(actual.tobytes(),source.resize((512,512),Image.Resampling.BILINEAR).resize((800,800),Image.Resampling.BILINEAR).tobytes())

    def test_gncnn_rejects_legacy_scale_and_unspecified_recipe(self):
        with self.assertRaises(ValueError):prepare_gncnn_input(Image.new('RGB',(2048,2048)),dict(input_recipe='author_qupath4096_resize800'))
        with self.assertRaises(ValueError):prepare_gncnn_input(Image.new('RGB',(4096,4096)),{})


if __name__=='__main__':unittest.main()
