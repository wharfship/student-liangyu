"""Encode VLM transport copies as JPEG without changing source pixel dimensions."""
import base64
import copy
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image

from .candidate_views import dump


def jpeg_message(text,paths,output):
    content=[dict(type='text',text=text)]
    for p in paths:
        with Image.open(p) as image:
            buffer=BytesIO();image.convert('RGB').save(buffer,format='JPEG',quality=95,subsampling=0)
        data=buffer.getvalue();digest=hashlib.sha256(data).hexdigest()
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        target=output/(digest+'.jpg')
        if not target.exists():target.write_bytes(data)
        content.append(dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(data).decode())))
    return dict(role='user',content=content)


def jpeg_transport(messages,output):
    """Historical reader may build PNG messages; only the converted copies are sent."""
    messages=copy.deepcopy(messages);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    receipt=[]
    for message in messages:
        if not isinstance(message.get('content'),list):continue
        for part in message['content']:
            if part.get('type')!='image_url':continue
            url=part['image_url']['url']
            if not url.startswith('data:image/'):raise ValueError('Expected local image data URL')
            source=base64.b64decode(url.split(',',1)[1])
            with Image.open(BytesIO(source)) as image:
                size=list(image.size);buffer=BytesIO()
                image.convert('RGB').save(buffer,format='JPEG',quality=95,subsampling=0)
            data=buffer.getvalue();digest=hashlib.sha256(data).hexdigest();path=output/(digest+'.jpg')
            if not path.exists():path.write_bytes(data)
            part['image_url']['url']='data:image/jpeg;base64,'+base64.b64encode(data).decode()
            receipt.append(dict(path=str(path.resolve()),size=size,sha256=digest,source_bytes=len(source),jpeg_bytes=len(data)))
    return messages,receipt


def read_candidate_jpeg(task,output,variant='autonomous'):
    from .visual_reader import api_call,read_candidate
    n=0
    def call(messages):
        nonlocal n
        n+=1;sent,receipt=jpeg_transport(messages,Path(output)/'transport')
        dump(Path(output)/f'transport_{n}_request.json',dict(messages=sent,images=receipt))
        return api_call(sent)
    return read_candidate(task,output,variant,caller=call)
