"""JPEG transport must preserve dimensions, sources and earlier messages."""
import base64
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from glomerulus_tools.catalog import sha
from glomerulus_tools.vlm_jpeg import jpeg_message,jpeg_transport


class JpegTests(unittest.TestCase):
    def test_transport_changes_copy_not_source_or_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png';Image.new('RGB',(151,93),(120,70,155)).save(source)
            digest=sha(source)
            url='data:image/png;base64,'+base64.b64encode(source.read_bytes()).decode()
            messages=[dict(role='user',content=[dict(type='image_url',image_url={'url':url})])]
            sent,receipts=jpeg_transport(messages,root/'transport')
            self.assertEqual(messages[0]['content'][0]['image_url']['url'],url)
            self.assertEqual(sha(source),digest);self.assertEqual(receipts[0]['size'],[151,93])
            actual=sent[0]['content'][0]['image_url']['url']
            self.assertTrue(actual.startswith('data:image/jpeg;base64,'))
            im=Image.open(BytesIO(base64.b64decode(actual.split(',',1)[1])))
            self.assertEqual(im.format,'JPEG');self.assertEqual(im.size,(151,93))
            self.assertEqual(Path(receipts[0]['path']).suffix,'.jpg')

    def test_message_is_always_jpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png';Image.new('RGBA',(41,29),(20,30,40,255)).save(source)
            message=jpeg_message('Read this candidate',[source],root/'transport')
            self.assertTrue(message['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,'))


if __name__=='__main__':unittest.main()
