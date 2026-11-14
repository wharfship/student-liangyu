"""Verify coordinate mapping and class identity in the streamed native exports."""
import tempfile
from pathlib import Path
import unittest
import cv2
import numpy as np
import tifffile
from glomerulus_tools.multiclass import export_native


class NativeSemanticExport(unittest.TestCase):
    def test_native_labels_and_independent_vessel_masks(self):
        a=np.array([[0,1,2,3],[3,2,1,0],[0,3,2,1]],dtype=np.uint8)
        expected=cv2.resize(a,(11,9),interpolation=cv2.INTER_NEAREST)
        with tempfile.TemporaryDirectory() as tmp:
            for cid in [None,2,3]:
                path=Path(tmp)/(str(cid)+'.tif')
                counts=export_native(a,[11,9],path,cid)
                actual=tifffile.imread(path)
                np.testing.assert_array_equal(actual,expected if cid is None else (expected==cid).astype(np.uint8)*255)
                self.assertEqual(counts,np.bincount(expected.ravel(),minlength=4).tolist())


if __name__=='__main__':unittest.main()
