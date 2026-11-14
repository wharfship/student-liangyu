import copy
import unittest

from PIL import Image
from glomerulus_tools.classification import validate_task,make_crop


class ClassificationContractTests(unittest.TestCase):
    def setUp(self):
        self.card={'species':['human'],'stains':['PAS']}
        self.task=dict(schema_version=1,species='human',target='candidate_identity',
            images=[dict(image_id='x',image='raw.png',image_sha256='a'*64,native_size=[300,200],
                         stain='PAS',source_mpp=.5,scale_basis='working_assumption',scale_evidence='explicit assumption')],
            candidates=[dict(candidate_id='c',image_id='x',box=[50,20,250,180])])

    def test_valid_task(self):
        self.assertEqual(list(validate_task(self.task,self.card)),['x'])

    def test_gt_and_scores_rejected_at_each_level(self):
        for container,key in [('root','gt'),('image','dice'),('candidate','proxy')]:
            task=copy.deepcopy(self.task)
            target=task if container=='root' else task['images' if container=='image' else 'candidates'][0]
            target[key]=1
            with self.assertRaises(ValueError):validate_task(task,self.card)

    def test_unsupported_stain_and_species(self):
        for key,value in [('species','mouse'),('stain','HE')]:
            task=copy.deepcopy(self.task)
            (task if key=='species' else task['images'][0])[key]=value
            with self.assertRaises(ValueError):validate_task(task,self.card)

    def test_invalid_mpp(self):
        for mpp in [None,0,-1,float('nan'),float('inf'),True]:
            self.task['images'][0]['source_mpp']=mpp
            with self.assertRaises(ValueError):validate_task(self.task,self.card)

    def test_missing_scale_evidence(self):
        self.task['images'][0]['scale_evidence']=''
        with self.assertRaises(ValueError):validate_task(self.task,self.card)

    def test_invalid_boxes(self):
        for box in [[0,0,301,200],[0,0,10.5,20],[10,10,10,20],[-1,0,10,20]]:
            self.task['candidates'][0]['box']=box
            with self.assertRaises(ValueError):validate_task(self.task,self.card)

    def test_duplicate_or_unlinked_candidates(self):
        self.task['candidates']*=2
        with self.assertRaises(ValueError):validate_task(self.task,self.card)
        self.task['candidates']=self.task['candidates'][:1]
        self.task['candidates'][0]['image_id']='unknown'
        with self.assertRaises(ValueError):validate_task(self.task,self.card)

    def test_complete_candidate_preserved_in_square(self):
        im=Image.new('RGB',(300,200),(100,50,75))
        crop,r=make_crop(im,[50,20,250,180],'plip')
        self.assertEqual(crop.size,(300,300))
        self.assertEqual(r['crop_box'],[0,0,300,200])
        self.assertEqual(r['white_padding'],[0,50,0,50])
        self.assertEqual(crop.getpixel((0,50)),(100,50,75))
        self.assertEqual(crop.getpixel((0,0)),(255,255,255))

    def test_author_glo_size_and_scale_receipt(self):
        crop,r=make_crop(Image.new('RGB',(300,200)),[50,20,250,180],'glo5')
        self.assertEqual(crop.size,(256,256))
        self.assertEqual(r['source_pixels_per_model_pixel'],[300/224,200/224])


if __name__=='__main__':unittest.main()
