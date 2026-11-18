import { FilesetResolver, PoseLandmarker, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const $ = id => document.getElementById(id);
const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");
let landmarker, stream, running=false, facing="environment", raf=0, lastTime=-1;
let lowSince=null, previousCenter=null, previousAt=0, lastFallAt=0, fallCount=0, frames=0, fpsStart=performance.now();
const drawing = () => new DrawingUtils(ctx);

function say(s){$("message").textContent=s}
function resize(){
 const r=canvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,1.5);
 const w=Math.round(r.width*d),h=Math.round(r.height*d);
 if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}
}
function logEvent(text){
 const li=document.createElement("li");li.textContent=`${new Date().toLocaleTimeString()} — ${text}`;
 const list=$("events");if(list.children.length===1&&list.textContent.includes("System ready"))list.innerHTML="";
 list.prepend(li);while(list.children.length>12)list.lastChild.remove();
}
async function loadModel(){
 if(landmarker)return;
 say("Loading pose AI model. First download may take a moment.");
 const vision=await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
 landmarker=await PoseLandmarker.createFromOptions(vision,{
  baseOptions:{modelAssetPath:"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",delegate:"GPU"},
  runningMode:"VIDEO",numPoses:1,minPoseDetectionConfidence:.55,minPosePresenceConfidence:.55,minTrackingConfidence:.55
 });
}
async function openCamera(){
 if(!navigator.mediaDevices?.getUserMedia)throw new Error("Camera requires Safari and a secure HTTPS website.");
 if(stream)stream.getTracks().forEach(t=>t.stop());
 stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:facing},width:{ideal:640},height:{ideal:480}}});
 video.srcObject=stream;video.style.display="block";await video.play();$("placeholder").style.display="none";resize();
}
const mid=(a,b)=>({x:(a.x+b.x)/2,y:(a.y+b.y)/2});
function visible(p){return p&&((p.visibility??1)>.45)}
function analyze(points,now){
 const ls=points[11],rs=points[12],lh=points[23],rh=points[24],la=points[27],ra=points[28];
 if(![ls,rs,lh,rh].every(visible))return null;
 const shoulder=mid(ls,rs),hip=mid(lh,rh);
 const angle=Math.abs(Math.atan2(hip.y-shoulder.y,hip.x-shoulder.x)*180/Math.PI);
 const center=mid(shoulder,hip);
 const shoulderWidth=Math.abs(ls.x-rs.x);
 const bodyLength=Math.hypot(hip.x-shoulder.x,hip.y-shoulder.y);
 const horizontal=angle<35 || angle>145;
 const low=shoulder.y>.43 && hip.y>.48 && bodyLength<Math.max(.25,shoulderWidth*1.5);
 const rapidDrop=previousCenter && (center.y-previousCenter.y)>.075 && (now-previousAt)<900;
 previousCenter=center;previousAt=now;
 let candidate=horizontal&&low;
 if(candidate){if(lowSince===null)lowSince=now;}else{lowSince=null}
 const sustained=lowSince!==null&&(now-lowSince)>1800;
 const possible=(rapidDrop&&candidate)||sustained;
 return {angle,center,possible,sustained,lowMs:lowSince===null?0:now-lowSince,feet:visible(la)||visible(ra)};
}
function render(result,now){
 ctx.clearRect(0,0,canvas.width,canvas.height);
 const pose=result.landmarks?.[0];
 if(!pose){$("person").textContent="NO";$("personHint").textContent="No full body visible";$("posture").textContent="UNKNOWN";$("postureHint").textContent="Move farther into camera view";$("angle").textContent="—";$("height").textContent="—";$("duration").textContent="0.0 s";$("alertBanner").hidden=true;lowSince=null;previousCenter=null;return}
 const util=drawing();
 util.drawConnectors(pose,PoseLandmarker.POSE_CONNECTIONS,{color:"#65f4e6",lineWidth:3});
 util.drawLandmarks(pose,{color:"#f2fffd",lineWidth:1,radius:3});
 const data=analyze(pose,now);
 $("person").textContent="YES";$("personHint").textContent="Body landmarks tracked";
 if(!data){$("posture").textContent="PARTIAL";$("postureHint").textContent="Show shoulders and hips";return}
 $("angle").textContent=`${data.angle.toFixed(0)}°`;
 $("height").textContent=`${Math.round(data.center.y*100)}%`;
 $("duration").textContent=`${(data.lowMs/1000).toFixed(1)} s`;
 const lying=data.angle<35||data.angle>145;
 $("posture").textContent=lying?"HORIZONTAL": "UPRIGHT / MOVING";
 $("postureHint").textContent=lying?"Horizontal body position observed":"No sustained horizontal posture";
 const banner=$("alertBanner");
 if(data.possible && now-lastFallAt>10000){
  lastFallAt=now;fallCount++;$("falls").textContent=String(fallCount);
  banner.hidden=false;logEvent("POSSIBLE FALL — check the person immediately");
  say("Possible fall pattern detected. This is an experimental alert; check the person directly.");
  setTimeout(()=>{banner.hidden=true},8000);
 }else if(!data.possible){banner.hidden=true}
}
function loop(){
 if(!running)return;
 if(video.readyState>=2&&video.currentTime!==lastTime){
  lastTime=video.currentTime;resize();
  try{const result=landmarker.detectForVideo(video,performance.now());render(result,performance.now())}
  catch(e){console.error(e);say("Frame processing error. Stop and restart the monitor.")}
  frames++;const now=performance.now();if(now-fpsStart>1000){$("fps").textContent=`${Math.round(frames*1000/(now-fpsStart))} FPS`;frames=0;fpsStart=now}
 }
 raf=requestAnimationFrame(loop);
}
async function start(){
 if(running)return;$("start").disabled=true;$("state").textContent="STARTING";
 try{await loadModel();await openCamera();running=true;$("stop").disabled=false;$("state").textContent="MONITORING";$("state").classList.add("active");say("Monitoring started. Keep the full body visible, with the camera fixed.");lastTime=-1;loop()}
 catch(e){console.error(e);say(e.name==="NotAllowedError"?"Camera permission denied. Allow camera access for Safari in iPhone Settings.":e.message||"Could not start camera or load AI model.");$("state").textContent="STANDBY";$("start").disabled=false}
}
function stop(){
 running=false;cancelAnimationFrame(raf);if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}
 video.pause();video.srcObject=null;video.style.display="none";ctx.clearRect(0,0,canvas.width,canvas.height);
 $("placeholder").style.display="flex";$("state").textContent="STANDBY";$("state").classList.remove("active");
 $("start").disabled=false;$("stop").disabled=true;$("alertBanner").hidden=true;
 $("person").textContent="NO";$("personHint").textContent="Monitoring stopped";$("posture").textContent="—";$("postureHint").textContent="No pose available";$("angle").textContent="—";$("height").textContent="—";$("duration").textContent="0.0 s";
 lowSince=null;previousCenter=null;say("Monitoring stopped. Camera is off; no video was saved.");
}
$("start").addEventListener("click",start);$("stop").addEventListener("click",stop);
$("flip").addEventListener("click",async()=>{facing=facing==="user"?"environment":"user";if(running){try{await openCamera();lastTime=-1;say("Camera switched.")}catch(e){say(e.message)}}else say(`Next camera: ${facing==="user"?"front":"rear"}. Tap Start Monitoring.`)});
$("clear").addEventListener("click",()=>{$("events").innerHTML="<li>Local history cleared.</li>";fallCount=0;$("falls").textContent="0"});
window.addEventListener("resize",resize);window.addEventListener("pagehide",stop);
