#![cfg(feature = "cuda")]
use std::sync::Arc;
use demucs_core_native::cuda_engine::{CudaState, GpuTensor};
fn max_diff(a:&[f32],b:&[f32])->f32{a.iter().zip(b).map(|(x,y)|(x-y).abs()).fold(0.0,f32::max)}

#[test]
#[ignore]
fn flatten_unflatten_bcft(){
    let state=Arc::new(CudaState::new(0).expect("c"));
    let b=1; let c=2; let f=3; let t=4;
    let in_:Vec<f32>=(0..b*c*f*t).map(|i|i as f32).collect();
    // CPU flatten: out[(ti*f+fri)*c+ci] = in[((ci)*f+fri)*t+ti]
    let mut cpu=vec![0.0f32;b*t*f*c];
    for ci in 0..c{for fri in 0..f{for ti in 0..t{cpu[(ti*f+fri)*c+ci]=in_[((ci)*f+fri)*t+ti];}}}
    let g=state.upload_f32(&in_,vec![b,c,f,t]).expect("up");
    let flat=demucs_core_native::cuda_ops::flatten_bcft_to_btfc(&state,&g).expect("flat");
    let dl=state.download_to_f32(&flat).expect("dl");
    eprintln!("flat cpu[0..6]={:?}",&cpu[0..6]);
    eprintln!("flat gpu[0..6]={:?}",&dl[0..6]);
    eprintln!("flatten max_diff={}",max_diff(&cpu,&dl));
    assert!(max_diff(&cpu,&dl)<1e-3,"flatten wrong");
    // unflatten round-trip
    let unf=demucs_core_native::cuda_ops::unflatten_btfc_to_bcft(&state,&flat,f,t).expect("unf");
    let dl2=state.download_to_f32(&unf).expect("dl2");
    eprintln!("roundtrip max_diff={}",max_diff(&in_,&dl2));
    assert!(max_diff(&in_,&dl2)<1e-3,"unflatten round-trip wrong");
}
