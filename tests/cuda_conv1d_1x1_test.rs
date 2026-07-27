#![cfg(feature = "cuda")]
use std::sync::Arc;
use demucs_core_native::cuda_engine::{CudaState, GpuTensor};
use demucs_core_native::gpu_model::{GpuBias, GpuConv1dWeight};
use demucs_core_native::model::{Bias, Conv1dWeight};
fn max_diff(a:&[f32],b:&[f32])->f32{a.iter().zip(b).map(|(x,y)|(x-y).abs()).fold(0.0,f32::max)}

#[test]
#[ignore]
fn conv1d_1x1_small(){
    let state=Arc::new(CudaState::new(0).expect("c"));
    let b=2; let cin=3; let l=4; let cout=5;
    // x [b, cin, l]
    let x:Vec<f32>=(0..b*cin*l).map(|i|((i as f32)*0.1-0.3)).collect();
    // w [cout, cin, 1]; Conv1dWeight.data row-major
    let w:Vec<f32>=(0..cout*cin).map(|i|((i as f32)*0.07-0.2)).collect();
    let bb_:Vec<f32>=(0..cout).map(|i|0.05*(i as f32)).collect();
    // CPU: y[b,co,l]=sum_ci x[b,ci,l]*w[co,ci]+bb[co]
    let mut cpu=vec![0.0f32;b*cout*l];
    for bi in 0..b{for co in 0..cout{for li in 0..l{let mut s=bb_[co];for ci in 0..cin{s+=x[(bi*cin+ci)*l+li]*w[co*cin+ci];}cpu[(bi*cout+co)*l+li]=s;}}}
    let gx=state.upload_f32(&x,vec![b,cin,l]).expect("x");
    let gw=GpuConv1dWeight::from_cpu(&state,&Conv1dWeight{data:w,out_ch:cout,in_ch:cin,k:1}).expect("w");
    let gb=GpuBias::from_cpu(&state,&Bias{data:bb_,len:cout}).expect("b");
    let gy:GpuTensor=demucs_core_native::cuda_ops::conv1d_1x1(&state,&gx,&gw,&gb).expect("y");
    let dl=state.download_to_f32(&gy).expect("dl");
    eprintln!("cpu[0..5]={:?}",&cpu[0..5]);
    eprintln!("gpu[0..5]={:?}",&dl[0..5]);
    eprintln!("conv1d_1x1 max_diff={:.4}",max_diff(&cpu,&dl));
}
