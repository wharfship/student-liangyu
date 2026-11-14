"""The reader must preserve coordinates, local budgets, and a no-edit role."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from glomerulus_tools.candidate_views import CandidateViews
from glomerulus_tools.catalog import sha
from glomerulus_tools.visual_reader import read_candidate, validate


class ReaderTests(unittest.TestCase):
    def fixture(self, root):
        pixels = np.zeros((31, 47, 3), dtype='uint8')
        pixels[:, :, 0] = np.arange(47)
        pixels[:, :, 1] = np.arange(31)[:, None]
        p = root/'image.png'; Image.fromarray(pixels).save(p)
        s = root/'support.npz'; np.savez(s, a=np.ones((10, 12), bool), b=np.zeros((10, 12), bool))
        return dict(candidate_id='test', image=str(p), image_sha256=sha(p), native_size=[47,31],
                    stain='PAS', source_mpp=.5, scale_basis='assumed', box=[7,9,19,19],
                    support_npz=str(s), support_sha256=sha(s))

    def test_native_view_preserves_source_and_outline_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=self.fixture(root);views=CandidateViews(task,root/'views')
            paths,receipt=views.render('native_center')
            raw=np.array(Image.open(paths[0]));marked=np.array(Image.open(paths[1]))
            self.assertEqual(receipt['source_box'],[0,0,47,31])
            self.assertEqual(receipt['scale'],[1,1])
            self.assertEqual(raw[9,7].tolist(),[7,9,0])
            self.assertEqual(marked[9,7].tolist(),[255,230,0])
            self.assertEqual(marked[12,12].tolist(),raw[12,12].tolist())
            with self.assertRaises(ValueError):views.render('native_center')
            with self.assertRaises(ValueError):views.render('../../image')
            with self.assertRaises(ValueError):CandidateViews(dict(task,gt='secret'),root/'bad')
            task['box']=[7,9,20,19]
            with self.assertRaisesRegex(ValueError,'shape'):CandidateViews(task,root/'bad')

    def test_local_agent_requests_then_submits_without_edit_permission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=self.fixture(root)
            cards=[dict(action='request_view',view='context',reason='Need surrounding epithelial arrangement'),
                   dict(action='submit_reading',identity='non_glomerulus',p_glomerulus=.1,adequacy='adequate',
                        differential='tubule',evidence='Single epithelial ring and lumen',uncertainty='')]
            def call(messages):
                return dict(seconds=.1,raw=dict(usage={},choices=[dict(message={'content':json.dumps(cards.pop(0))})]))
            result=read_candidate(task,root/'run',caller=call)
            self.assertEqual(result['view_requests'],1);self.assertEqual(result['api_calls'],2)
            self.assertFalse(result['edit_authorized']);self.assertFalse(result['boundary_assessed'])
            self.assertEqual(result['identity'],'non_glomerulus')
            self.assertNotIn('evidence',result)

    def test_invalid_or_exhausted_action_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=self.fixture(root)
            def call(messages):
                return dict(seconds=.1,raw=dict(choices=[dict(message={'content':json.dumps(dict(
                    action='request_view',view='context',reason='need context'))})]))
            result=read_candidate(task,root/'run',variant='pathology_static',caller=call)
            self.assertEqual(result['status'],'failed');self.assertTrue(result['review_required'])
            self.assertEqual(result['api_calls'],1)
            for card in [dict(action='edit_mask'),dict(action='request_view',view='unknown',reason='x')]:
                with self.assertRaises(ValueError):validate(card,True)


if __name__=='__main__':unittest.main()
