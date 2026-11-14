"""Query-chunked SAM attention: same weights, grid and relative-position formula."""
import importlib
import types

def forward(self,x):
    import torch
    b,h,w,_=x.shape
    qkv=self.qkv(x).reshape(b,h*w,3,self.num_heads,-1).permute(2,0,3,1,4)
    q,k,v=qkv.reshape(3,b*self.num_heads,h*w,-1).unbind(0)
    if self.use_rel_pos:
        get_rel_pos=importlib.import_module(type(self).__module__).get_rel_pos
        rh=get_rel_pos(h,h,self.rel_pos_h);rw=get_rel_pos(w,w,self.rel_pos_w)
        rq=q.reshape(b*self.num_heads,h,w,-1)
        rel_h=torch.einsum('bhwc,hkc->bhwk',rq,rh)
        rel_w=torch.einsum('bhwc,wkc->bhwk',rq,rw)
    parts=[]
    for start in range(0,h*w,256):
        end=min(start+256,h*w);a=(q[:,start:end]*self.scale)@k.transpose(-2,-1)
        if self.use_rel_pos:
            pos=torch.arange(start,end,device=x.device);yy=pos//w;xx=pos%w
            a=(a.reshape(b*self.num_heads,end-start,h,w)+rel_h[:,yy,xx,:,None]+rel_w[:,yy,xx,None,:]).flatten(-2)
        parts.append(a.softmax(dim=-1)@v)
    z=torch.cat(parts,dim=1).reshape(b,self.num_heads,h,w,-1).permute(0,2,3,1,4).reshape(b,h,w,-1)
    return self.proj(z)

def install(encoder):
    count=0
    for block in encoder.blocks:
        if block.window_size==0:
            block.attn.forward=types.MethodType(forward,block.attn);count+=1
    return count

def verify(attention_class,device='cpu'):
    """Compare to official attention on a grid that exercises multiple chunks."""
    import torch
    torch.manual_seed(20261111)
    a=attention_class(dim=64,num_heads=4,qkv_bias=True,use_rel_pos=True,input_size=(17,19)).to(device).eval()
    # Nonzero positional terms ensure the comparison tests relative-position math.
    torch.nn.init.normal_(a.rel_pos_h,std=.1);torch.nn.init.normal_(a.rel_pos_w,std=.1)
    x=torch.randn(2,17,19,64,device=device)
    with torch.inference_mode():
        original=a(x);chunked=forward(a,x)
    error=float((original-chunked).abs().max());assert torch.allclose(original,chunked,atol=2e-6,rtol=2e-5),error
    return dict(max_abs_error=error,grid=[17,19],dtype='float32',nonzero_relative_position=True)
