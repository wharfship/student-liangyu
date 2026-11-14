"""Meaningful contract failures: leakage, scale, unsupported dispatch and native geometry."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from glomerulus_tools.catalog import get_card, load_registry
from glomerulus_tools.contract import bind_tool_scale, compatibility, prepare_task, read_wsi_header, validate_task


class ContractTests(unittest.TestCase):
    def task(self,**changes):
        return dict(dict(task_id='a',image='unused.png',species='human',stain='PAS',
                         target='whole_glomerular_unit',source_mpp=.5,scale_basis='working_assumption'),**changes)

    def test_online_tasks_reject_offline_and_nonfinite_scale(self):
        for key in ('gt','ground_truth','dice','baseline_mask'):
            with self.assertRaisesRegex(ValueError,'offline'):validate_task(self.task(**{key:'x'}))
        for mpp in (True,0,-1,float('inf'),float('nan')):
            with self.assertRaisesRegex(ValueError,'MPP'):validate_task(self.task(source_mpp=mpp))

    def test_aliases_do_not_infer_input_stain(self):
        self.assertEqual(validate_task(self.task(stain='H&E',input_form='ROI'))['stain'],'HE')
        self.assertEqual(validate_task(self.task(stain='PAM'))['stain'],'SIL')
        self.assertIn('unsupported_stain',compatibility(get_card('nepath_segformer_b5'),self.task(stain='UNKNOWN'),True))

    def test_research_does_not_override_species_target_or_disabled(self):
        card=get_card('nepath_segformer_b5')
        self.assertIn('unsupported_species',compatibility(card,self.task(species='mouse'),True))
        self.assertIn('target_mismatch',compatibility(card,self.task(target='tuft'),True))
        self.assertIn('tool_disabled',compatibility(get_card('mori_seg_paused'),self.task(),True))
        silver=self.task(stain='SIL')
        self.assertIn('exploratory_stain',compatibility(card,silver))
        self.assertFalse(compatibility(card,silver,True))

    def test_scale_evidence_and_grid_not_silently_guessed(self):
        task=self.task();task.pop('scale_basis')
        self.assertIn('missing_scale_evidence',compatibility(get_card('nepath_segformer_b5'),task))
        self.assertIn('unverified_native_grid',compatibility(get_card('likiseg333_light'),self.task(source_mpp=.25),True))

    def test_actual_roi_size_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rgb.png';Image.new('RGB',(31,19)).save(path)
            task=self.task(image=str(path))
            self.assertEqual(prepare_task(task)['native_size'],[31,19])
            with self.assertRaisesRegex(ValueError,'native_size'):prepare_task(dict(task,native_size=[19,31]))
            task.pop('scale_basis')
            with self.assertRaisesRegex(ValueError,'source_mpp'):prepare_task(task)

    def test_ome_units_and_header_conflicts(self):
        import tifffile
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rgb.ome.tif'
            tifffile.imwrite(path,np.zeros((19,31,3),np.uint8),photometric='rgb',ome=True,
                metadata={'axes':'YXS','PhysicalSizeX':250,'PhysicalSizeXUnit':'nm','PhysicalSizeY':250,'PhysicalSizeYUnit':'nm'})
            header=read_wsi_header(path)
            self.assertEqual(header['native_size'],[31,19]);self.assertEqual(header['x'],.25)
            task=self.task(image=str(path),input_form='wsi',source_mpp=.25)
            result=prepare_task(task)
            self.assertEqual(result['scale_basis'],'measured_metadata')
            with self.assertRaisesRegex(ValueError,'conflicts'):prepare_task(dict(task,source_mpp=.5))

    def test_catalog_identity_and_no_scores_in_runtime_registry(self):
        data=load_registry();card=get_card('segformer_b5')
        self.assertNotIn('researcher_note',json.dumps(data,ensure_ascii=False))
        self.assertNotIn('额外可分割动脉和小动脉',json.dumps(data,ensure_ascii=False))
        self.assertEqual(card['id'],'nepath_segformer_b5')
        with self.assertRaises(ValueError):get_card('saltfish_kpis')
        for stain in ['he','pas','sil','tri']:
            self.assertFalse(get_card('ki2021_'+stain+'_unit_batchstats')['enabled'])
        self.assertEqual(get_card('likiseg333_light')['environment']['nnUNet_compile'],'false')
        for card in data['tools']:
            self.assertNotIn('results',card)
            self.assertNotIn('dice',card)

    def test_binary_and_native_coordinates_are_checked(self):
        from glomerulus_tools.adapters import finalize_record
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mask.png';Image.fromarray(np.ones((7,9),np.uint8)).save(path)
            task=self.task(native_size=[9,7]);card=get_card('nepath_segformer_b5')
            with self.assertRaisesRegex(ValueError,'binary'):finalize_record(card,task,Path(tmp),{'mask_path':str(path)})
            Image.fromarray(np.full((7,9),255,np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError,'native'):finalize_record(card,dict(task,native_size=[7,9]),Path(tmp),{'mask_path':str(path)})

    def test_legacy_scaling_is_derived_once_and_conflicts_rejected(self):
        task=self.task(source_mpp=.5)
        result=bind_tool_scale(get_card('omni_full'),task)
        self.assertEqual(result['per_tool_downsample']['omni_cap'],4.)
        self.assertEqual(bind_tool_scale(get_card('ki2021_pas_unit_batchstats'),task)['per_tool_downsample']['ki2021'],3.84)
        with self.assertRaisesRegex(ValueError,'conflicts'):
            bind_tool_scale(get_card('omni_full'),dict(task,per_tool_downsample={'omni_cap':8}))

    def test_glo_native_and_virtual_wsi_are_distinct_recipes(self):
        task=self.task(input_form='wsi',source_mpp=.65)
        self.assertIn('unsupported_glo_crop_mpp',compatibility(get_card('glo2_whole'),task))
        self.assertFalse(compatibility(get_card('glo2_wsi_virtual05'),task))
        self.assertIn('unsupported_input_form',compatibility(get_card('glo2_wsi_virtual05'),self.task()))


if __name__=='__main__':unittest.main()
