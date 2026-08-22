// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "http_server.h"

#include <httplib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "audio_file.h"
#include "audio_resampler.h"
#include "engine_registry.h"
#include "json.h"
#include "subtitles.h"

namespace nemo_speech::http {
namespace {
using json::Value;

constexpr const char* kPlaygroundParts[] = {
    R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeMo-Speech.cpp</title><style>
:root{color-scheme:dark;--bg:#101417;--card:#1a2025;--line:#303940;--accent:#76b900;--muted:#a8b1b8}*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:#f4f7f8;font:15px system-ui,sans-serif}main{max-width:920px;margin:auto;padding:32px 20px}
h1{font-size:28px;margin:0 0 6px}p{color:var(--muted)}nav{display:flex;flex-wrap:wrap;gap:8px;margin:24px 0}button,.button{border:1px solid var(--line);background:#242c32;color:#fff;padding:10px 14px;border-radius:7px;cursor:pointer}
button.active,button.primary{background:var(--accent);border-color:var(--accent);color:#101410;font-weight:650}.panel{display:none;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px}.panel.active{display:block}
label{display:block;color:var(--muted);margin:12px 0 6px}input,textarea,select{width:100%;border:1px solid var(--line);background:#101417;color:#fff;padding:10px;border-radius:6px}input[type=file]{border-style:dashed;padding:18px}input[type=file].drag{border-color:var(--accent);background:#17200f}textarea{min-height:120px;resize:vertical}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}.actions{display:flex;gap:8px;margin-top:14px;align-items:center;flex-wrap:wrap}.check{display:flex;align-items:center;gap:8px;color:#f4f7f8}.check input{width:auto}.source-switch{display:flex;gap:8px;margin:10px 0}.source-switch button{flex:1}[hidden]{display:none!important}pre{white-space:pre-wrap;background:#0c0f11;border-radius:7px;padding:14px;min-height:52px}.status{color:var(--muted)}.server-status{margin:14px 0}.models{color:var(--muted);margin-top:8px}.models ul{margin:6px 0}audio{width:100%;margin-top:14px}.timeline{position:relative;min-height:56px;margin-top:14px;background:#0c0f11;border-radius:7px;overflow:hidden}.segment{position:absolute;height:18px;border-radius:3px;padding:1px 4px;overflow:hidden;white-space:nowrap;font-size:11px;color:#101417}@media(max-width:620px){.row{grid-template-columns:1fr}.source-switch{flex-direction:column}}
</style></head><body><main><h1>NeMo-Speech.cpp</h1><p>Local speech inference. Audio stays on this machine.</p><div id="server-status" class="server-status status">Connecting…</div><details class="models"><summary>Loaded models</summary><ul id="model-list"></ul></details>
<nav><button class="tab active" data-tab="asr" data-capability="transcription">Transcribe</button><button class="tab" data-tab="diar" data-capability="diarization">Diarize</button><button class="tab" data-tab="tts" data-capability="speech">Synthesize</button><button class="tab" data-tab="nmt" data-capability="translation">Translate</button><button class="tab" data-tab="speech-translate" data-capability="speech-translation">Speech translate</button></nav>
<section id="asr" class="panel active"><label>Choose or drop WAV audio</label><input id="audio-file" type="file" accept="audio/wav,.wav"><div class="row"><div><label>Language (optional)</label><input id="asr-lang" placeholder="en-US"></div><div><label>Response</label><select id="asr-format"><option value="verbose_json">JSON with timestamps</option><option value="json">JSON text</option><option value="text">Plain text</option><option value="srt">SRT subtitles</option><option value="vtt">WebVTT subtitles</option></select></div></div><label id="asr-speakers-option" class="check" hidden><input id="asr-speakers" type="checkbox">Identify speakers in the transcript</label><div class="actions"><button class="primary" id="transcribe">Transcribe file</button><button id="listen">Use microphone</button><button id="stop-listening" disabled>Stop</button><button id="download-asr" disabled>Download</button><span class="status" id="asr-status"></span></div><div id="asr-timeline" class="timeline"></div><pre id="asr-output"></pre></section>
<section id="diar" class="panel"><label>Choose or drop WAV audio</label><input id="diar-file" type="file" accept="audio/wav,.wav"><div class="actions"><button class="primary" id="diarize">Identify speakers</button><button id="download-diar-json" disabled>Download JSON</button><button id="download-diar-rttm" disabled>Download RTTM</button><span class="status" id="diar-status"></span></div><div id="diar-timeline" class="timeline"></div><pre id="diar-output"></pre></section>
<section id="tts" class="panel"><label>Text</label><textarea id="tts-text" placeholder="What would you like me to say?"></textarea><div class="row"><div><label>Language</label><select id="tts-lang"><option value="en-US">English (US)</option></select></div><div><label>Voice</label><select id="tts-voice"><option value="">Default</option></select></div></div><div class="actions"><button class="primary" id="speak">Synthesize</button><button id="download-tts" disabled>Download WAV</button><span class="status" id="tts-status"></span></div><audio id="tts-audio" controls></audio></section>
<section id="nmt" class="panel"><label>Text</label><textarea id="nmt-text"></textarea><div class="row"><div><label>From</label><input id="nmt-from" value="en-US"></div><div><label>To</label><input id="nmt-to" value="es-US"></div></div><div class="actions"><button class="primary" id="translate">Translate</button><span class="status" id="nmt-status"></span></div><pre id="nmt-output"></pre></section>
<section id="speech-translate" class="panel"><label>Source</label><div class="source-switch"><button class="speech-source active" data-source="file">Audio file</button><button class="speech-source" data-source="microphone">Microphone</button><button class="speech-source" data-source="text">Text</button></div><div data-speech-source="file"><label>Choose or drop WAV audio</label><input id="speech-translate-file" type="file" accept="audio/wav,.wav"></div><div data-speech-source="microphone" hidden><div class="actions"><button id="speech-translate-record">Start recording</button><button id="speech-translate-stop" disabled>Stop recording</button><span class="status" id="speech-translate-record-status">No recording yet.</span></div></div><div data-speech-source="text" hidden><label>Text to translate</label><textarea id="speech-translate-input" placeholder="Enter source text"></textarea></div><div class="row"><div><label>From <span id="speech-translate-from-hint">(optional)</span></label><input id="speech-translate-from" placeholder="auto"></div><div><label>To</label><input id="speech-translate-to" value="es-US"></div></div><div class="actions"><button class="primary" id="speech-translate-text">Translate</button><button id="speech-translate-audio" data-capability="speech-to-speech">Translate and speak</button><button id="download-speech-translate" disabled>Download</button><span class="status" id="speech-translate-status"></span></div><audio id="speech-translate-player" controls></audio><pre id="speech-translate-output"></pre></section>
)HTML",
    R"HTML(<script>
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab,.panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.tab).classList.add('active')});
const busy=(id,on,text='Working…')=>{const e=document.getElementById(id);e.textContent=on?text:''};
const downloads={};const downloadable=(id,name,data,type)=>{if(downloads[id])URL.revokeObjectURL(downloads[id]);const url=URL.createObjectURL(data instanceof Blob?data:new Blob([data],{type}));downloads[id]=url;const button=document.getElementById(id);button.disabled=false;button.onclick=()=>{const a=document.createElement('a');a.href=url;a.download=name;a.click()}};
const timeline=(id,segments)=>{const view=document.getElementById(id),end=Math.max(.001,...segments.map(x=>x.end||0));view.replaceChildren();segments.forEach((x,i)=>{const speaker=x.speaker||0,b=document.createElement('div');b.className='segment';b.textContent=speaker?`S${speaker}`:(x.word||'speech');b.title=`${speaker?'Speaker '+speaker+': ':''}${(x.start||0).toFixed(2)}–${(x.end||0).toFixed(2)} s${x.word?' '+x.word:''}`;b.style.left=`${100*(x.start||0)/end}%`;b.style.width=`${Math.max(.6,100*((x.end||0)-(x.start||0))/end)}%`;b.style.top=`${4+(speaker?((speaker-1)%3)*18:0)}px`;b.style.background=speaker?`hsl(${(speaker*83)%360} 65% 62%)`:'var(--accent)';view.appendChild(b)})};
const speakerTranscript=(words,fallback='')=>{if(!words?.some(x=>x.speaker))return fallback;const lines=[];let speaker=0,text='';const flush=()=>{if(text)lines.push(`Speaker ${speaker||'?'}: ${text}`);text=''};words.forEach(x=>{const next=x.speaker||0;if(text&&next!==speaker)flush();speaker=next;const word=x.word||'';text=!text?word:/^[,.;:!?%\)\]\}]/.test(word)?text+word:`${text} ${word}`});flush();return lines.join('\n')};
const wavBlob=(chunks,sampleRate)=>{const samples=chunks.reduce((n,x)=>n+x.length,0),buffer=new ArrayBuffer(44+samples*2),view=new DataView(buffer),write=(at,text)=>{for(let i=0;i<text.length;i++)view.setUint8(at+i,text.charCodeAt(i))};write(0,'RIFF');view.setUint32(4,36+samples*2,true);write(8,'WAVE');write(12,'fmt ');view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);view.setUint32(24,sampleRate,true);view.setUint32(28,sampleRate*2,true);view.setUint16(32,2,true);view.setUint16(34,16,true);write(36,'data');view.setUint32(40,samples*2,true);let offset=44;chunks.forEach(chunk=>{for(const sample of chunk){view.setInt16(offset,sample,true);offset+=2}});return new Blob([buffer],{type:'audio/wav'})};
const apiKey=new URLSearchParams(location.search).get('api_key'),apiFetch=(url,options={})=>{if(apiKey)options.headers={...(options.headers||{}),Authorization:`Bearer ${apiKey}`};return fetch(url,options)};
document.querySelectorAll('input[type=file]').forEach(input=>{input.ondragover=e=>{e.preventDefault();input.classList.add('drag')};input.ondragleave=()=>input.classList.remove('drag');input.ondrop=e=>{e.preventDefault();input.classList.remove('drag');if(e.dataTransfer.files.length)input.files=e.dataTransfer.files}});
Promise.all([apiFetch('/ready').then(r=>r.json()),apiFetch('/version').then(r=>r.json()),apiFetch('/v1/models').then(r=>{if(!r.ok)throw Error('model inventory requires an API key');return r.json()})]).then(([ready,version,inventory])=>{const models=inventory.data||[],capabilities=new Set(models.map(x=>x.capability)),names=models.map(x=>`${x.id} (${x.capability})`);document.getElementById('server-status').textContent=`Ready · v${version.version} · device ${ready.device||'auto'} · ${ready.capabilities.length} capabilities`;const list=document.getElementById('model-list');list.replaceChildren(...names.map(name=>{const item=document.createElement('li');item.textContent=name;return item}));document.querySelectorAll('[data-capability]').forEach(b=>{if(!capabilities.has(b.dataset.capability)){b.disabled=true;b.title=`Load a ${b.dataset.capability} model to enable this panel`}});document.getElementById('asr-speakers-option').hidden=!capabilities.has('diarization');const tts=models.find(x=>x.capability==='speech');if(tts){const voice=document.getElementById('tts-voice');voice.replaceChildren(new Option('Default',''),...(tts.voices||[]).map(x=>new Option(x,x)));const names={'en-US':'English (US)','es-ES':'Spanish (Spain)','de-DE':'German','fr-FR':'French','it-IT':'Italian','vi-VN':'Vietnamese','hi-IN':'Hindi','zh-CN':'Mandarin Chinese','ja-JP':'Japanese'},language=document.getElementById('tts-lang');language.replaceChildren(...(tts.languages||['en-US']).map(x=>new Option(names[x]||x,x)))}const active=document.querySelector('.tab.active');if(active?.disabled)document.querySelector('.tab:not(:disabled)')?.click()}).catch(e=>{document.getElementById('server-status').textContent=`Server unavailable: ${e.message}`});
const asrSpeakers=document.getElementById('asr-speakers'),asrFormat=document.getElementById('asr-format');asrSpeakers.onchange=()=>{if(asrSpeakers.checked)asrFormat.value='verbose_json'};asrFormat.onchange=()=>{if(asrFormat.value!=='verbose_json')asrSpeakers.checked=false};
document.getElementById('transcribe').onclick=async()=>{const f=document.getElementById('audio-file').files[0];if(!f)return alert('Choose a WAV file');const d=new FormData(),format=document.getElementById('asr-format').value,speakers=asrSpeakers.checked;d.append('file',f);d.append('response_format',format);if(speakers)d.append('diarization','true');const l=document.getElementById('asr-lang').value;if(l)d.append('language',l);busy('asr-status',1);try{const r=await apiFetch('/v1/audio/transcriptions',{method:'POST',body:d});const t=await r.text();if(!r.ok)throw Error(t);const parsed=format==='verbose_json'?JSON.parse(t):null,words=parsed?.words||[];document.getElementById('asr-output').textContent=speakers?speakerTranscript(words,parsed?.text||''):t;timeline('asr-timeline',words);const ext=format==='verbose_json'||format==='json'?'json':format==='text'?'txt':format;downloadable('download-asr',`transcript.${ext}`,t,r.headers.get('content-type')||'text/plain')}catch(e){document.getElementById('asr-output').textContent=e}finally{busy('asr-status',0)}};
document.getElementById('diarize').onclick=async()=>{const f=document.getElementById('diar-file').files[0];if(!f)return alert('Choose a WAV file');const d=new FormData();d.append('file',f);busy('diar-status',1);try{const r=await apiFetch('/v1/audio/diarizations',{method:'POST',body:d}),t=await r.text();if(!r.ok)throw Error(t);const j=JSON.parse(t),segments=j.segments||[],pretty=JSON.stringify(j,null,2),recording=f.name.replace(/\.[^.]+$/,'').replace(/\s+/g,'_'),rttm=segments.map(x=>`SPEAKER ${recording} 1 ${x.start.toFixed(3)} ${(x.end-x.start).toFixed(3)} <NA> <NA> speaker_${x.speaker} <NA> <NA>`).join('\n')+'\n';timeline('diar-timeline',segments);document.getElementById('diar-output').textContent=pretty;downloadable('download-diar-json','diarization.json',pretty,'application/json');downloadable('download-diar-rttm','diarization.rttm',rttm,'text/plain')}catch(e){document.getElementById('diar-output').textContent=e}finally{busy('diar-status',0)}};
)HTML",
    R"HTML(let live={};const liveOut=document.getElementById('asr-output'),listen=document.getElementById('listen'),stopListen=document.getElementById('stop-listening');
const stopCapture=async()=>{live.processor?.disconnect();live.source?.disconnect();live.media?.getTracks().forEach(t=>t.stop());if(live.context?.state!=='closed')await live.context.close()};
const resetLive=()=>{listen.disabled=false;stopListen.disabled=true;busy('asr-status',0)};
listen.onclick=async()=>{try{const media=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true},video:false}),context=new AudioContext(),source=context.createMediaStreamSource(media),processor=context.createScriptProcessor(4096,1,1),speakers=asrSpeakers.checked,scheme=location.protocol==='https:'?'wss':'ws',key=new URLSearchParams(location.search).get('api_key'),url=`${scheme}://${location.host}/v1/realtime${key?'?api_key='+encodeURIComponent(key):''}`,socket=new WebSocket(url);socket.binaryType='arraybuffer';const render=()=>{const final=(live.finals||[]).join('\n');liveOut.textContent=final+(final&&live.partial?'\n':'')+(live.partial||'')};socket.onopen=()=>socket.send(JSON.stringify({type:'session.update',session:{sample_rate:context.sampleRate,language:document.getElementById('asr-lang').value,automatic_punctuation:true,word_timestamps:speakers,speaker_diarization:speakers}}));socket.onmessage=e=>{const m=JSON.parse(e.data);if(m.type.endsWith('.delta')){live.partial=(live.partial||'')+(m.delta||'');render()}else if(m.type.endsWith('.completed')){live.finals.push(speakers?speakerTranscript(m.words,m.transcript||''):(m.transcript||''));live.partial='';render();if(m.words)timeline('asr-timeline',m.words)}else if(m.type==='input_audio_buffer.committed'){socket.close()}else if(m.type==='error'){liveOut.textContent=m.error.message;if(live.stopping)socket.close()}};socket.onclose=resetLive;processor.onaudioprocess=e=>{if(socket.readyState!==1||live.stopping)return;const f=e.inputBuffer.getChannelData(0),p=new Int16Array(f.length);for(let i=0;i<f.length;i++)p[i]=Math.max(-32768,Math.min(32767,Math.round(f[i]*32767)));socket.send(p.buffer)};source.connect(processor);processor.connect(context.destination);live={media,context,source,processor,socket,finals:[],partial:'',stopping:false};listen.disabled=true;stopListen.disabled=false;busy('asr-status',1,'Listening…')}catch(e){liveOut.textContent=e;resetLive()}};
stopListen.onclick=async()=>{if(live.stopping)return;live.stopping=true;stopListen.disabled=true;busy('asr-status',1,'Finalizing…');await stopCapture();if(live.socket?.readyState===1)live.socket.send(JSON.stringify({type:'input_audio_buffer.commit'}));else{live.socket?.close();resetLive()}};
document.getElementById('speak').onclick=async()=>{busy('tts-status',1);try{const r=await apiFetch('/v1/audio/speech',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({input:document.getElementById('tts-text').value,language:document.getElementById('tts-lang').value,voice:document.getElementById('tts-voice').value,response_format:'wav'})});if(!r.ok)throw Error(await r.text());const audio=await r.blob();downloadable('download-tts','speech.wav',audio,'audio/wav');document.getElementById('tts-audio').src=downloads['download-tts']}catch(e){alert(e)}finally{busy('tts-status',0)}};
document.getElementById('translate').onclick=async()=>{busy('nmt-status',1);try{const r=await apiFetch('/v1/translations',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({input:document.getElementById('nmt-text').value,source_language:document.getElementById('nmt-from').value,target_language:document.getElementById('nmt-to').value})});const t=await r.text();if(!r.ok)throw Error(t);const j=JSON.parse(t);document.getElementById('nmt-output').textContent=j.translations.map(x=>x.text).join('\n')}catch(e){document.getElementById('nmt-output').textContent=e}finally{busy('nmt-status',0)}};
let speechSource='file',speechMic={};const speechRecord=document.getElementById('speech-translate-record'),speechStop=document.getElementById('speech-translate-stop'),speechRecordStatus=document.getElementById('speech-translate-record-status');
document.querySelectorAll('.speech-source').forEach(button=>button.onclick=()=>{if(speechMic.recording)return;speechSource=button.dataset.source;document.querySelectorAll('.speech-source').forEach(x=>x.classList.toggle('active',x===button));document.querySelectorAll('[data-speech-source]').forEach(x=>x.hidden=x.dataset.speechSource!==speechSource);const text=speechSource==='text';document.getElementById('speech-translate-from-hint').textContent=text?'(required)':'(optional)';document.getElementById('speech-translate-from').placeholder=text?'en-US':'auto'});
const stopSpeechRecording=async()=>{if(!speechMic.recording)return;speechMic.recording=false;speechMic.processor.disconnect();speechMic.source.disconnect();speechMic.sink.disconnect();speechMic.media.getTracks().forEach(track=>track.stop());await speechMic.context.close();speechMic.blob=wavBlob(speechMic.chunks,speechMic.sampleRate);const seconds=speechMic.chunks.reduce((n,x)=>n+x.length,0)/speechMic.sampleRate;speechRecord.disabled=false;speechStop.disabled=true;speechRecordStatus.textContent=`Recording ready · ${seconds.toFixed(1)} s`};
speechRecord.onclick=async()=>{try{const media=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true},video:false}),context=new AudioContext(),source=context.createMediaStreamSource(media),processor=context.createScriptProcessor(4096,1,1),sink=context.createGain(),chunks=[];sink.gain.value=0;processor.onaudioprocess=e=>{const input=e.inputBuffer.getChannelData(0),pcm=new Int16Array(input.length);for(let i=0;i<input.length;i++)pcm[i]=Math.max(-32768,Math.min(32767,Math.round(input[i]*32767)));chunks.push(pcm)};source.connect(processor);processor.connect(sink);sink.connect(context.destination);speechMic={media,context,source,processor,sink,chunks,sampleRate:context.sampleRate,recording:true};speechRecord.disabled=true;speechStop.disabled=false;speechRecordStatus.textContent='Recording…'}catch(e){speechRecordStatus.textContent=e}};
speechStop.onclick=stopSpeechRecording;
const speechTranslate=async speak=>{const out=document.getElementById('speech-translate-output'),sourceLanguage=document.getElementById('speech-translate-from').value.trim(),targetLanguage=document.getElementById('speech-translate-to').value.trim();if(!targetLanguage)return alert('Choose a target language');if(speechMic.recording)return alert('Stop the microphone recording first');busy('speech-translate-status',1);try{if(speechSource==='text'){const input=document.getElementById('speech-translate-input').value.trim();if(!input)return alert('Enter text to translate');if(!sourceLanguage)return alert('Source language is required for text translation');const translated=await apiFetch('/v1/translations',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({input,source_language:sourceLanguage,target_language:targetLanguage})}),body=await translated.text();if(!translated.ok)throw Error(body);const text=JSON.parse(body).translations.map(x=>x.text).join('\n');if(!speak){out.textContent=text;downloadable('download-speech-translate','translation.txt',text,'text/plain');return}const response=await apiFetch('/v1/audio/speech',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({input:text,language:targetLanguage,response_format:'wav'})});if(!response.ok)throw Error(await response.text());const audio=await response.blob();downloadable('download-speech-translate','translated-speech.wav',audio,'audio/wav');document.getElementById('speech-translate-player').src=downloads['download-speech-translate'];out.textContent=text;return}const audio=speechSource==='microphone'?speechMic.blob:document.getElementById('speech-translate-file').files[0];if(!audio)return alert(speechSource==='microphone'?'Record microphone audio first':'Choose a WAV file');const d=new FormData();d.append('file',audio,speechSource==='microphone'?'microphone.wav':audio.name);d.append('language',sourceLanguage);d.append('target_language',targetLanguage);const path=speak?'/v1/audio/speech/translations':'/v1/audio/translations';d.append('response_format',speak?'wav':'verbose_json');const r=await apiFetch(path,{method:'POST',body:d});if(!r.ok)throw Error(await r.text());if(speak){const translatedAudio=await r.blob();downloadable('download-speech-translate','translated-speech.wav',translatedAudio,'audio/wav');document.getElementById('speech-translate-player').src=downloads['download-speech-translate'];out.textContent='Translated speech is ready.'}else{const text=await r.text();out.textContent=JSON.stringify(JSON.parse(text),null,2);downloadable('download-speech-translate','translation.json',out.textContent,'application/json')}}catch(e){out.textContent=e}finally{busy('speech-translate-status',0)}};
document.getElementById('speech-translate-text').onclick=()=>speechTranslate(false);document.getElementById('speech-translate-audio').onclick=()=>speechTranslate(true);
</script></main></body></html>)HTML"};

std::string
error_body(const std::string& message, const std::string& code = "invalid_request_error") {
    Value root(Value::Object{});
    Value error(Value::Object{});
    error["message"] = message;
    error["type"] = code;
    error["param"] = Value();
    error["code"] = Value();
    root["error"] = std::move(error);
    return root.dump();
}

#if defined(NEMO_SPEECH_REGISTRY_TTS)
std::string
openai_voice(const tts::Synthesizer& synthesizer, std::string requested) {
    std::string normalized = requested;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (normalized.empty() || normalized == "default")
        return {};
    for (const auto& model_voice : synthesizer.speaker_names()) {
        std::string candidate = model_voice;
        std::transform(candidate.begin(), candidate.end(), candidate.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (candidate == normalized)
            return model_voice;
    }
    static constexpr std::array<const char*, 11> standard_voices = {
        "alloy", "ash",  "ballad", "coral",   "echo", "fable",
        "nova",  "onyx", "sage",   "shimmer", "verse"};
    if (std::find(standard_voices.begin(), standard_voices.end(), normalized) !=
        standard_voices.end())
        return {};
    return requested;
}
#endif

void
fail(
    httplib::Response& response, int status, const std::string& message,
    const std::string& code = "invalid_request_error") {
    response.status = status;
    response.set_content(error_body(message, code), "application/json");
}

#if defined(NEMO_SPEECH_REGISTRY_ASR)
std::string
form_value(
    const httplib::Request& request, const std::string& name, const std::string& fallback = {}) {
    if (request.form.has_field(name))
        return request.form.get_field(name);
    if (request.has_param(name))
        return request.get_param_value(name);
    return fallback;
}

// `speech_contexts` - [{"phrases": ["..."], "boost": N}] - parsed into
// recognition options. Same field name and shape as the gRPC RecognitionConfig
// so word boosting reads identically across surfaces.
void
apply_speech_contexts(const Value& parsed, std::vector<asr::AsrRequestOptions::Boost>& out) {
    if (!parsed.is_array())
        throw std::invalid_argument("speech_contexts must be a JSON array");
    for (const Value& context : parsed.array()) {
        asr::AsrRequestOptions::Boost boost;
        boost.boost = static_cast<float>(context.number_or("boost", 10.0));
        if (const Value* phrases = context.find("phrases")) {
            if (!phrases->is_array())
                throw std::invalid_argument("speech_contexts[].phrases must be an array");
            for (const Value& phrase : phrases->array()) boost.phrases.push_back(phrase.string());
        }
        if (!boost.phrases.empty())
            out.push_back(std::move(boost));
    }
}

void
apply_speech_contexts(const std::string& json, std::vector<asr::AsrRequestOptions::Boost>& out) {
    if (json.empty())
        return;
    try {
        apply_speech_contexts(Value::parse(json), out);
    }
    catch (const std::invalid_argument&) {
        throw;
    }
    catch (const std::exception&) {
        throw std::invalid_argument("speech_contexts must be valid JSON");
    }
}

bool
form_bool(const httplib::Request& request, const std::string& name, bool fallback = false) {
    std::string value = form_value(request, name);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value.empty())
        return fallback;
    if (value == "true" || value == "1" || value == "yes" || value == "on")
        return true;
    if (value == "false" || value == "0" || value == "no" || value == "off")
        return false;
    throw std::invalid_argument(name + " must be a boolean");
}
#endif

#if defined(NEMO_SPEECH_REGISTRY_ASR) || defined(NEMO_SPEECH_REGISTRY_DIAR)
httplib::FormData
uploaded_file(const httplib::Request& request) {
    if (!request.form.has_file("file"))
        throw std::invalid_argument("multipart field 'file' is required");
    return request.form.get_file("file");
}
#endif

#if defined(NEMO_SPEECH_REGISTRY_ASR)
std::string
timecode(int milliseconds, bool vtt) {
    milliseconds = std::max(0, milliseconds);
    char result[32];
    std::snprintf(
        result, sizeof(result), "%02d:%02d:%02d%c%03d", milliseconds / 3600000,
        milliseconds / 60000 % 60, milliseconds / 1000 % 60, vtt ? '.' : ',', milliseconds % 1000);
    return result;
}
#endif

#if defined(NEMO_SPEECH_REGISTRY_ASR)
std::string
decode_base64(const std::string& input) {
    static constexpr int8_t invalid = -1;
    static const auto table = [] {
        std::array<int8_t, 256> result{};
        result.fill(invalid);
        const std::string alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (size_t i = 0; i < alphabet.size(); ++i)
            result[static_cast<unsigned char>(alphabet[i])] = static_cast<int8_t>(i);
        return result;
    }();
    std::string output;
    output.reserve(input.size() * 3 / 4);
    uint32_t accumulator = 0;
    int bits = 0;
    size_t sextets = 0;
    size_t padding = 0;
    for (const unsigned char c : input) {
        if (c == '=') {
            if (++padding > 2)
                throw std::invalid_argument("audio is not valid base64");
            continue;
        }
        if (std::isspace(c))
            continue;
        if (padding != 0 || table[c] == invalid)
            throw std::invalid_argument("audio is not valid base64");
        accumulator = (accumulator << 6) | static_cast<uint32_t>(table[c]);
        ++sextets;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            output.push_back(static_cast<char>((accumulator >> bits) & 0xff));
        }
    }
    const size_t remainder = sextets % 4;
    const size_t expected_padding = remainder == 2 ? 2 : remainder == 3 ? 1 : 0;
    if (remainder == 1 || (padding != 0 && padding != expected_padding) ||
        (bits != 0 && (accumulator & ((uint32_t{1} << bits) - 1)) != 0))
        throw std::invalid_argument("audio is not valid base64");
    return output;
}

std::vector<float>
pcm16_samples(const std::string& bytes) {
    if (bytes.size() % 2 != 0)
        throw std::invalid_argument("PCM16 audio frames must contain a whole number of samples");
    std::vector<float> output(bytes.size() / 2);
    for (size_t i = 0; i < output.size(); ++i) {
        const auto low = static_cast<uint8_t>(bytes[i * 2]);
        const auto high = static_cast<uint8_t>(bytes[i * 2 + 1]);
        const int16_t value =
            static_cast<int16_t>(static_cast<uint16_t>(low) | (static_cast<uint16_t>(high) << 8));
        output[i] = value / 32768.0f;
    }
    return output;
}

std::string
transcript_response(
    const asr::Result& result, const std::string& requested_language, const std::string& format) {
    const asr::Alternative empty;
    const auto& alternative = result.alternatives.empty() ? empty : result.alternatives.front();
    if (format == "text")
        return alternative.transcript + "\n";
    if (format == "srt" || format == "vtt") {
        const bool vtt = format == "vtt";
        int audio_ms = std::max(1, static_cast<int>(std::round(result.audio_processed * 1000)));
        std::vector<subtitle::Word> words;
        words.reserve(alternative.words.size());
        for (const auto& word : alternative.words)
            words.push_back(
                {word.word, word.start_time, word.end_time, word.confidence, word.speaker_tag});
        const auto cues = subtitle::make_cues(words, alternative.transcript, audio_ms);
        std::string output = vtt ? "WEBVTT\n\n" : "";
        for (size_t i = 0; i < cues.size(); ++i) {
            if (!vtt)
                output += std::to_string(i + 1) + "\n";
            output += timecode(cues[i].start_ms, vtt) + " --> " +
                      timecode(std::max(cues[i].start_ms + 1, cues[i].end_ms), vtt) + "\n";
            output += cues[i].text + "\n\n";
        }
        return output;
    }
    Value root(Value::Object{});
    root["text"] = alternative.transcript;
    if (format == "verbose_json") {
        root["task"] = "transcribe";
        root["duration"] = result.audio_processed;
        root["language"] = !alternative.language_codes.empty() ? alternative.language_codes.front()
                                                               : requested_language;
        Value::Array words;
        for (const auto& word : alternative.words) {
            Value item(Value::Object{});
            item["word"] = word.word;
            item["start"] = word.start_time / 1000.0;
            item["end"] = word.end_time / 1000.0;
            item["confidence"] = word.confidence;
            if (word.speaker_tag > 0)
                item["speaker"] = word.speaker_tag;
            words.emplace_back(std::move(item));
        }
        root["words"] = std::move(words);
    }
    return root.dump();
}
#endif

}  // namespace

struct Server::Impl {
    EngineRegistry& models;
    ServerConfig config;
    std::unique_ptr<httplib::Server> server;
    std::atomic<uint64_t> request_id{1};

    Impl(EngineRegistry& engines, ServerConfig config)
        : models(engines), config(std::move(config)) {
        const bool has_cert = !this->config.tls_certificate.empty();
        const bool has_key = !this->config.tls_private_key.empty();
        if (has_cert != has_key)
            throw std::invalid_argument("TLS requires both a certificate and private key");
        if (has_cert) {
#if defined(CPPHTTPLIB_OPENSSL_SUPPORT)
            server = std::make_unique<httplib::SSLServer>(
                this->config.tls_certificate.c_str(), this->config.tls_private_key.c_str());
            if (!server->is_valid())
                throw std::runtime_error("could not initialize the TLS server");
#else
            throw std::invalid_argument(
                "TLS support is not enabled; rebuild with -DNEMO_SPEECH_HTTP_TLS=ON");
#endif
        } else {
            server = std::make_unique<httplib::Server>();
        }
        server->set_payload_max_length(this->config.max_upload_bytes);
        server->set_read_timeout(this->config.read_timeout_seconds);
        server->set_write_timeout(this->config.write_timeout_seconds);
        const int threads = std::max(1, this->config.threads);
        server->new_task_queue = [threads] { return new httplib::ThreadPool(threads); };
        httplib::Headers default_headers{
            {"Access-Control-Allow-Headers", "Authorization, Content-Type"},
            {"Content-Security-Policy",
             "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
             "media-src 'self' blob:; connect-src 'self' ws: wss:; object-src 'none'; "
             "base-uri 'none'; frame-ancestors 'none'"},
            {"X-Content-Type-Options", "nosniff"},
        };
        if (!this->config.cors_origin.empty())
            default_headers.emplace("Access-Control-Allow-Origin", this->config.cors_origin);
        server->set_default_headers(std::move(default_headers));
        if (this->config.access_log) {
            server->set_logger(
                [this](const httplib::Request& request, const httplib::Response& response) {
                    if (this->config.json_logs) {
                        Value event(Value::Object{});
                        event["event"] = "http.request";
                        event["request_id"] = response.get_header_value("X-Request-Id");
                        event["method"] = request.method;
                        event["path"] = request.path;
                        event["status"] = response.status;
                        event["remote_address"] = request.remote_addr;
                        std::fprintf(stderr, "%s\n", event.dump().c_str());
                    } else {
                        std::fprintf(
                            stderr, "http request_id=%s method=%s path=%s status=%d remote=%s\n",
                            response.get_header_value("X-Request-Id").c_str(),
                            request.method.c_str(), request.path.c_str(), response.status,
                            request.remote_addr.c_str());
                    }
                });
        }
        server->set_pre_routing_handler(
            [this](const httplib::Request& request, httplib::Response& response) {
                response.set_header("X-Request-Id", std::to_string(request_id.fetch_add(1)));
                if (this->config.api_key.empty() || request.method == "OPTIONS" ||
                    request.path == "/" || request.path == "/health" || request.path == "/ready" ||
                    request.path == "/version")
                    return httplib::Server::HandlerResponse::Unhandled;
                const std::string expected = "Bearer " + this->config.api_key;
                const bool is_realtime = request.path == "/v1/realtime" ||
                                         request.path == "/v1/audio/transcriptions/realtime";
                const bool query_authorized =
                    is_realtime && request.has_param("api_key") &&
                    request.get_param_value("api_key") == this->config.api_key;
                if (request.get_header_value("Authorization") != expected && !query_authorized) {
                    fail(response, 401, "missing or invalid bearer token");
                    return httplib::Server::HandlerResponse::Handled;
                }
                return httplib::Server::HandlerResponse::Unhandled;
            });
        server->Options("/.*", [](const httplib::Request&, httplib::Response& response) {
            response.status = 204;
            response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        });
        server->set_error_handler([](const httplib::Request& request, httplib::Response& response) {
            if (request.path.rfind("/v1", 0) != 0 || !response.body.empty())
                return;
            const int status = response.status > 0 ? response.status : 500;
            fail(
                response, status, status == 404 ? "API route not found" : "HTTP request failed",
                status >= 500 ? "server_error" : "invalid_request_error");
        });
        server->set_exception_handler(
            [](const httplib::Request&, httplib::Response& response, std::exception_ptr error) {
                try {
                    if (error)
                        std::rethrow_exception(error);
                }
                catch (const std::invalid_argument& exception) {
                    fail(response, 400, exception.what());
                    return;
                }
                catch (const std::exception& exception) {
                    fail(response, 500, exception.what(), "server_error");
                    return;
                }
                fail(response, 500, "unknown server error", "server_error");
            });

        server->Get("/", [this](const httplib::Request&, httplib::Response& response) {
            if (!this->config.enable_playground) {
                fail(response, 404, "the browser playground is disabled");
                return;
            }
            std::string playground;
            for (const char* part : kPlaygroundParts) playground += part;
            response.set_content(std::move(playground), "text/html; charset=utf-8");
        });
        server->Get("/health", [this](const httplib::Request&, httplib::Response& response) {
            Value body(Value::Object{});
            body["status"] = this->models.ready() ? "ok" : "not_ready";
            body["version"] = NEMO_SPEECH_VERSION_STR;
            response.status = this->models.ready() ? 200 : 503;
            response.set_content(body.dump(), "application/json");
        });
        server->Get("/ready", [this](const httplib::Request&, httplib::Response& response) {
            Value body(Value::Object{});
            body["ready"] = this->models.ready();
            body["device"] = this->models.device_label();
            Value::Array capabilities;
            for (const auto& capability : this->models.capabilities())
                capabilities.emplace_back(capability);
            body["capabilities"] = std::move(capabilities);
            response.status = this->models.ready() ? 200 : 503;
            response.set_content(body.dump(), "application/json");
        });
        server->Get("/version", [](const httplib::Request&, httplib::Response& response) {
            Value body(Value::Object{});
            body["version"] = NEMO_SPEECH_VERSION_STR;
            response.set_content(body.dump(), "application/json");
        });
        server->Get("/v1/models", [this](const httplib::Request&, httplib::Response& response) {
            Value root(Value::Object{});
            root["object"] = "list";
            Value::Array data;
#if defined(NEMO_SPEECH_REGISTRY_ASR)
            try {
                auto model = this->models.asr();
                Value item(Value::Object{});
                item["id"] = model->model_name();
                item["object"] = "model";
                item["owned_by"] = "local";
                item["capability"] = "transcription";
                item["device"] = this->models.device_label();
                data.emplace_back(std::move(item));
            }
            catch (const std::exception&) {
            }
#endif
#if defined(NEMO_SPEECH_REGISTRY_DIAR)
            try {
                (void)this->models.diarization();
                Value item(Value::Object{});
                item["id"] = "diarization";
                item["object"] = "model";
                item["owned_by"] = "local";
                item["capability"] = "diarization";
                item["device"] = this->models.device_label();
                data.emplace_back(std::move(item));
            }
            catch (const std::exception&) {
            }
#endif
#if defined(NEMO_SPEECH_REGISTRY_TTS)
            try {
                auto model = this->models.tts();
                Value item(Value::Object{});
                item["id"] = model->model_name();
                item["object"] = "model";
                item["owned_by"] = "local";
                item["capability"] = "speech";
                item["device"] = this->models.device_label();
                Value::Array voices;
                for (const auto& voice : model->speaker_names()) voices.emplace_back(voice);
                item["voices"] = std::move(voices);
                Value::Array languages;
                for (const auto& language : model->supported_language_codes())
                    languages.emplace_back(language);
                item["languages"] = std::move(languages);
                data.emplace_back(std::move(item));
            }
            catch (const std::exception&) {
            }
#endif
#if defined(NEMO_SPEECH_REGISTRY_NMT)
            try {
                auto model = this->models.nmt();
                Value item(Value::Object{});
                item["id"] = model->model_name();
                item["object"] = "model";
                item["owned_by"] = "local";
                item["capability"] = "translation";
                item["device"] = this->models.device_label();
                data.emplace_back(std::move(item));
            }
            catch (const std::exception&) {
            }
#endif
            const auto capabilities = this->models.capabilities();
            const auto add_composition = [&](const char* name, const char* capability) {
                if (std::find(capabilities.begin(), capabilities.end(), name) == capabilities.end())
                    return;
                Value item(Value::Object{});
                item["id"] = name;
                item["object"] = "model";
                item["owned_by"] = "local";
                item["capability"] = capability;
                item["device"] = this->models.device_label();
                data.emplace_back(std::move(item));
            };
            add_composition("speech-translation", "speech-translation");
            add_composition("speech-to-speech", "speech-to-speech");
            root["data"] = std::move(data);
            response.set_content(root.dump(), "application/json");
        });

        server->Post(
            "/v1/audio/transcriptions",
            [this](const httplib::Request& request, httplib::Response& response) {
#if defined(NEMO_SPEECH_REGISTRY_ASR)
                const auto upload = uploaded_file(request);
                const auto audio = audio::load_wav_memory(
                    upload.content.data(), upload.content.size(), upload.filename);
                const std::string language = form_value(request, "language");
                const std::string format = form_value(request, "response_format", "json");
                if (format != "json" && format != "verbose_json" && format != "text" &&
                    format != "srt" && format != "vtt")
                    throw std::invalid_argument(
                        "response_format must be json, verbose_json, text, srt, or vtt");
                asr::AsrRequestOptions options;
                options.language_code = language;
                options.enable_automatic_punctuation =
                    form_bool(request, "automatic_punctuation", true);
                options.verbatim_transcripts = form_bool(request, "verbatim");
                options.profanity_filter = form_bool(request, "profanity_filter");
                options.enable_word_time_offsets =
                    format == "verbose_json" || format == "srt" || format == "vtt";
                options.enable_speaker_diarization = form_bool(request, "diarization");
                if (options.enable_speaker_diarization) {
                    const std::string speakers = form_value(request, "max_speaker_count");
                    if (!speakers.empty()) {
                        size_t consumed = 0;
                        int parsed = 0;
                        try {
                            parsed = std::stoi(speakers, &consumed);
                        }
                        catch (const std::exception&) {
                            consumed = 0;
                        }
                        if (consumed != speakers.size() || parsed <= 0)
                            throw std::invalid_argument(
                                "max_speaker_count must be a positive integer");
                        options.max_speaker_count = parsed;
                    }
                }
                if (options.enable_speaker_diarization && format != "verbose_json")
                    throw std::invalid_argument(
                        "diarization requires response_format=verbose_json");
                // OpenAI-compat shim: `prompt` = one boosted phrase at the default
                // score. The native field is `speech_contexts` (same shape as gRPC).
                const std::string prompt = form_value(request, "prompt");
                if (!prompt.empty())
                    options.speech_contexts.push_back({{prompt}, 10.0f});
                apply_speech_contexts(
                    form_value(request, "speech_contexts"), options.speech_contexts);
                auto recognizer = this->models.asr();
                const auto result = recognizer->recognize(
                    audio.samples.data(), audio.samples.size(), options, language,
                    audio.sample_rate);
                const auto body = transcript_response(result, language, format);
                const char* content_type =
                    format == "json" || format == "verbose_json"
                        ? "application/json"
                        : (format == "vtt" ? "text/vtt" : "text/plain; charset=utf-8");
                response.set_content(body, content_type);
#else
                (void)request;
                fail(response, 501, "this build does not include ASR");
#endif
            });

        server->Post(
            "/v1/audio/speech",
            [this](const httplib::Request& request, httplib::Response& response) {
#if defined(NEMO_SPEECH_REGISTRY_TTS)
                const Value body = Value::parse(request.body);
                auto synthesizer = this->models.tts();
                tts::SynthesisRequest synthesis;
                synthesis.text = body.string_or("input");
                synthesis.language_code = body.string_or("language");
                synthesis.voice_name = openai_voice(*synthesizer, body.string_or("voice"));
                if (const auto* sample_rate = body.find("sample_rate")) {
                    if (!sample_rate->is_number())
                        throw std::invalid_argument("sample_rate must be a positive integer");
                    const double requested_sample_rate = sample_rate->number();
                    if (requested_sample_rate != std::floor(requested_sample_rate))
                        throw std::invalid_argument("sample_rate must be a positive integer");
                    if (requested_sample_rate < 1 || requested_sample_rate > 96000)
                        throw std::invalid_argument("sample_rate must be between 1 and 96000");
                    synthesis.output_sample_rate = static_cast<int>(requested_sample_rate);
                }
                if (synthesis.text.empty())
                    throw std::invalid_argument("input is required");
                const double speed = body.number_or("speed", 1.0);
                if (std::abs(speed - 1.0) > 1e-9)
                    throw std::invalid_argument(
                        "this model does not support changing speech speed");
                const std::string format = body.string_or("response_format", "wav");
                if (format != "wav" && format != "pcm")
                    throw std::invalid_argument("response_format must be wav or pcm");
                std::string pcm;
                const auto result =
                    synthesizer->synthesize(synthesis, [&](const auto&, const std::string& chunk) {
                        pcm += chunk;
                        return true;
                    });
                if (format == "pcm")
                    response.set_content(std::move(pcm), "audio/pcm");
                else
                    response.set_content(
                        audio::pcm16_wav(pcm, result.metadata.sample_rate), "audio/wav");
#else
                (void)request;
                fail(response, 501, "this build does not include TTS");
#endif
            });

        server->Post(
            "/v1/audio/translations",
            [this](const httplib::Request& request, httplib::Response& response) {
#if defined(NEMO_SPEECH_REGISTRY_SPEECH)
                const auto upload = uploaded_file(request);
                const auto source_audio = audio::load_wav_memory(
                    upload.content.data(), upload.content.size(), upload.filename);
                const std::string requested_source = form_value(request, "language");
                const std::string target = form_value(request, "target_language", "en-US");
                const std::string format = form_value(request, "response_format", "json");
                if (format != "json" && format != "verbose_json" && format != "text")
                    throw std::invalid_argument(
                        "response_format must be json, verbose_json, or text");
                speech::SpeechTranslationOptions options;
                options.source_language = requested_source;
                options.target_language = target;
                options.recognition.language_code = requested_source;
                options.recognition.enable_automatic_punctuation =
                    form_bool(request, "automatic_punctuation", true);
                options.recognition.verbatim_transcripts = form_bool(request, "verbatim");
                options.recognition.profanity_filter = form_bool(request, "profanity_filter");
                const std::string prompt = form_value(request, "prompt");
                if (!prompt.empty())
                    options.recognition.speech_contexts.push_back({{prompt}, 10.0f});
                apply_speech_contexts(
                    form_value(request, "speech_contexts"), options.recognition.speech_contexts);
                std::vector<speech::SpeechTranslationResult> translated;
                speech::SpeechTranslationCallbacks callbacks;
                callbacks.translation = [&](const auto& result) {
                    translated.push_back(result);
                    return true;
                };
                auto stream = this->models.speech_translation()->streaming_translate(
                    std::move(options), std::move(callbacks));
                stream->push(
                    source_audio.samples.data(), source_audio.samples.size(),
                    source_audio.sample_rate);
                stream->finish();
                if (translated.empty())
                    throw std::runtime_error("NMT returned no translation");
                std::string text;
                for (const auto& result : translated) {
                    if (!text.empty() && !result.text.empty())
                        text += ' ';
                    text += result.text;
                }
                if (format == "text") {
                    response.set_content(text + "\n", "text/plain; charset=utf-8");
                    return;
                }
                Value output(Value::Object{});
                output["text"] = text;
                if (format == "verbose_json") {
                    output["task"] = "translate";
                    output["language"] = translated.back().language_code;
                    output["duration"] =
                        static_cast<double>(source_audio.samples.size()) / source_audio.sample_rate;
                }
                response.set_content(output.dump(), "application/json");
#else
                (void)request;
                fail(response, 501, "audio translation requires ASR and NMT in this build");
#endif
            });

        server->Post(
            "/v1/audio/speech/translations",
            [this](const httplib::Request& request, httplib::Response& response) {
#if defined(NEMO_SPEECH_REGISTRY_SPEECH) && defined(NEMO_SPEECH_REGISTRY_TTS)
                const auto upload = uploaded_file(request);
                const auto source_audio = audio::load_wav_memory(
                    upload.content.data(), upload.content.size(), upload.filename);
                speech::SpeechTranslationOptions options;
                options.source_language = form_value(request, "language");
                options.target_language = form_value(request, "target_language");
                options.synthesize_speech = true;
                options.recognition.language_code = options.source_language;
                options.recognition.enable_automatic_punctuation =
                    form_bool(request, "automatic_punctuation", true);
                options.recognition.verbatim_transcripts = form_bool(request, "verbatim");
                options.recognition.profanity_filter = form_bool(request, "profanity_filter");
                options.synthesis.language_code = options.target_language;
                options.synthesis.voice_name =
                    openai_voice(*this->models.tts(), form_value(request, "voice"));
                const std::string sample_rate = form_value(request, "sample_rate");
                if (!sample_rate.empty()) {
                    if (!std::all_of(sample_rate.begin(), sample_rate.end(), [](unsigned char c) {
                            return std::isdigit(c) != 0;
                        }))
                        throw std::invalid_argument("sample_rate must be a positive integer");
                    try {
                        const auto rate = std::stoul(sample_rate);
                        if (rate == 0 || rate > 96000)
                            throw std::invalid_argument("sample_rate must be between 1 and 96000");
                        options.synthesis.output_sample_rate = static_cast<int>(rate);
                    }
                    catch (const std::out_of_range&) {
                        throw std::invalid_argument("sample_rate must be between 1 and 96000");
                    }
                }
                const std::string format = form_value(request, "response_format", "wav");
                if (format != "wav" && format != "pcm")
                    throw std::invalid_argument("response_format must be wav or pcm");
                const std::string prompt = form_value(request, "prompt");
                if (!prompt.empty())
                    options.recognition.speech_contexts.push_back({{prompt}, 10.0f});
                apply_speech_contexts(
                    form_value(request, "speech_contexts"), options.recognition.speech_contexts);
                std::string pcm;
                int output_sample_rate = 0;
                speech::SpeechTranslationCallbacks callbacks;
                callbacks.audio = [&](const auto&, const auto& metadata, const std::string& chunk) {
                    if (output_sample_rate != 0 && output_sample_rate != metadata.sample_rate)
                        throw std::runtime_error("TTS changed sample rate within a response");
                    output_sample_rate = metadata.sample_rate;
                    pcm += chunk;
                    return true;
                };
                auto stream = this->models.speech_translation()->streaming_translate(
                    std::move(options), std::move(callbacks));
                stream->push(
                    source_audio.samples.data(), source_audio.samples.size(),
                    source_audio.sample_rate);
                stream->finish();
                if (pcm.empty() || output_sample_rate <= 0)
                    throw std::runtime_error("speech translation produced no audio");
                if (format == "pcm")
                    response.set_content(std::move(pcm), "audio/pcm");
                else
                    response.set_content(audio::pcm16_wav(pcm, output_sample_rate), "audio/wav");
#else
                (void)request;
                fail(response, 501, "speech-to-speech translation requires ASR, NMT, and TTS");
#endif
            });

        server->Post(
            "/v1/translations",
            [this](const httplib::Request& request, httplib::Response& response) {
#if defined(NEMO_SPEECH_REGISTRY_NMT)
                const Value body = Value::parse(request.body);
                std::vector<std::string> inputs;
                if (const auto* input = body.find("input")) {
                    if (input->is_string())
                        inputs.push_back(input->string());
                    else if (input->is_array())
                        for (const auto& value : input->array()) inputs.push_back(value.string());
                    else
                        throw std::invalid_argument("input must be a string or array of strings");
                }
                if (inputs.empty())
                    throw std::invalid_argument("input is required");
                const std::string source = body.string_or("source_language");
                const std::string target = body.string_or("target_language");
                if (source.empty() || target.empty())
                    throw std::invalid_argument("source_language and target_language are required");
                const auto translations = this->models.nmt()->translate(inputs, source, target);
                Value root(Value::Object{});
                Value::Array output;
                for (const auto& translation : translations) {
                    Value item(Value::Object{});
                    item["text"] = translation.text;
                    output.emplace_back(std::move(item));
                }
                root["translations"] = std::move(output);
                response.set_content(root.dump(), "application/json");
#else
                (void)request;
                fail(response, 501, "this build does not include translation");
#endif
            });

        auto diarize = [this](const httplib::Request& request, httplib::Response& response) {
#if defined(NEMO_SPEECH_REGISTRY_DIAR)
            const auto upload = uploaded_file(request);
            const auto source = audio::load_wav_memory(
                upload.content.data(), upload.content.size(), upload.filename);
            const std::string mode =
                request.form.has_field("mode") ? request.form.get_field("mode") : "streaming";
            if (mode != "streaming" && mode != "offline")
                throw std::invalid_argument("mode must be streaming or offline");
            auto engine = this->models.diarization();
            const auto result = engine->diarize(
                source.samples.data(), source.samples.size(), source.sample_rate,
                mode == "offline" ? asr::DiarizationMode::Offline
                                  : asr::DiarizationMode::Streaming);
            Value root(Value::Object{});
            Value::Array output;
            for (const auto& segment : result.segments) {
                Value item(Value::Object{});
                item["start"] = segment.t0;
                item["end"] = segment.t1;
                item["speaker"] = segment.speaker + 1;
                output.emplace_back(std::move(item));
            }
            root["segments"] = std::move(output);
            response.set_content(root.dump(), "application/json");
#else
            (void)request;
            fail(response, 501, "this build does not include diarization");
#endif
        };
        server->Post("/v1/diarizations", diarize);
        server->Post("/v1/audio/diarizations", std::move(diarize));

#if defined(NEMO_SPEECH_REGISTRY_ASR)
        auto realtime = [this](const httplib::Request& request, httplib::ws::WebSocket& socket) {
            if (!this->config.api_key.empty()) {
                const std::string expected = "Bearer " + this->config.api_key;
                const bool authorized =
                    request.get_header_value("Authorization") == expected ||
                    (request.has_param("api_key") &&
                     request.get_param_value("api_key") == this->config.api_key);
                if (!authorized) {
                    socket.close(httplib::ws::CloseStatus::PolicyViolation, "invalid bearer token");
                    return;
                }
            }
            auto recognizer = this->models.asr();
            asr::AsrRequestOptions options;
            options.enable_automatic_punctuation = true;
            int sample_rate = recognizer->sample_rate();
            std::unique_ptr<asr::RecognitionStream> stream;
            std::string partial_transcript;
            size_t audio_bytes = 0;
            auto ensure_stream = [&] {
                if (!stream)
                    stream = recognizer->streaming_recognize(
                        options, options.language_code, /*coordinate_ingress=*/true);
            };
            auto send = [&](Value event) {
                if (!event.find("event_id"))
                    event["event_id"] = "event_" + std::to_string(request_id.fetch_add(1));
                return socket.send(event.dump());
            };
            auto emit = [&](const asr::Result& result) {
                if (result.alternatives.empty())
                    return true;
                const auto& alternative = result.alternatives.front();
                Value event(Value::Object{});
                event["type"] = result.is_final
                                    ? "conversation.item.input_audio_transcription.completed"
                                    : "conversation.item.input_audio_transcription.delta";
                if (result.is_final) {
                    event["transcript"] = alternative.transcript;
                    partial_transcript.clear();
                } else {
                    const bool extends_previous =
                        alternative.transcript.rfind(partial_transcript, 0) == 0;
                    event["delta"] = extends_previous
                                         ? alternative.transcript.substr(partial_transcript.size())
                                         : alternative.transcript;
                    partial_transcript = alternative.transcript;
                }
                event["audio_processed"] = result.audio_processed;
                if (result.is_final && options.needs_word_timings()) {
                    Value::Array words;
                    for (const auto& word : alternative.words) {
                        Value item(Value::Object{});
                        item["word"] = word.word;
                        item["start"] = word.start_time / 1000.0;
                        item["end"] = word.end_time / 1000.0;
                        item["confidence"] = word.confidence;
                        if (word.speaker_tag > 0)
                            item["speaker"] = word.speaker_tag;
                        words.emplace_back(std::move(item));
                    }
                    event["words"] = std::move(words);
                }
                return send(std::move(event));
            };
            auto append_audio = [&](const std::string& pcm) {
                if (pcm.size() > this->config.max_upload_bytes -
                                     std::min(audio_bytes, this->config.max_upload_bytes))
                    throw std::invalid_argument(
                        "realtime audio exceeds the configured upload limit");
                audio_bytes += pcm.size();
                const auto samples = pcm16_samples(pcm);
                if (samples.empty())
                    return true;
                ensure_stream();
                stream->push(samples.data(), samples.size(), sample_rate);
                while (auto result = stream->next()) {
                    if (!emit(*result))
                        return false;
                    if (!result->is_final)
                        break;
                }
                return true;
            };
            Value created(Value::Object{});
            created["type"] = "session.created";
            Value session(Value::Object{});
            session["input_audio_format"] = "pcm16";
            session["sample_rate"] = sample_rate;
            session["model"] = recognizer->model_name();
            created["session"] = std::move(session);
            if (!send(std::move(created)))
                return;

            std::string message;
            for (;;) {
                const auto kind = socket.read(message);
                if (kind == httplib::ws::ReadResult::Fail)
                    break;
                try {
                    if (kind == httplib::ws::ReadResult::Binary) {
                        if (!append_audio(message))
                            break;
                        continue;
                    }
                    const Value event = Value::parse(message);
                    const std::string type = event.string_or("type");
                    if (type == "session.update" || type == "config") {
                        if (stream)
                            throw std::invalid_argument(
                                "session configuration cannot change after audio starts");
                        const Value* updated = event.find("session");
                        if (!updated)
                            updated = &event;
                        sample_rate =
                            static_cast<int>(updated->number_or("sample_rate", sample_rate));
                        if (!audio::supported_input_sample_rate(sample_rate))
                            throw std::invalid_argument(
                                "sample_rate must be between 8000 and 96000");
                        options.language_code =
                            updated->string_or("language", options.language_code);
                        options.enable_automatic_punctuation =
                            updated->bool_or("automatic_punctuation", true);
                        options.verbatim_transcripts = updated->bool_or("verbatim", false);
                        options.enable_word_time_offsets =
                            updated->bool_or("word_timestamps", false);
                        options.enable_speaker_diarization =
                            updated->bool_or("speaker_diarization", false);
                        const double speakers =
                            updated->number_or("max_speaker_count", options.max_speaker_count);
                        if (speakers <= 0.0 || speakers != std::floor(speakers))
                            throw std::invalid_argument(
                                "max_speaker_count must be a positive integer");
                        options.max_speaker_count = static_cast<int>(speakers);
                        options.profanity_filter = updated->bool_or("profanity_filter", false);
                        options.stop_history_eou_ms = static_cast<float>(
                            updated->number_or("endpointing_ms", options.stop_history_eou_ms));
                        options.speech_contexts.clear();
                        const std::string prompt = updated->string_or("prompt", "");
                        if (!prompt.empty())
                            options.speech_contexts.push_back({{prompt}, 10.0f});
                        if (const Value* contexts = updated->find("speech_contexts"))
                            apply_speech_contexts(*contexts, options.speech_contexts);
                        Value response(Value::Object{});
                        response["type"] = "session.updated";
                        response["session"] = *updated;
                        if (!send(std::move(response)))
                            break;
                    } else if (type == "input_audio_buffer.append") {
                        if (!append_audio(decode_base64(event.string_or("audio"))))
                            break;
                    } else if (type == "input_audio_buffer.commit") {
                        ensure_stream();
                        if (!emit(stream->finish()))
                            break;
                        stream.reset();
                        audio_bytes = 0;
                        Value committed(Value::Object{});
                        committed["type"] = "input_audio_buffer.committed";
                        if (!send(std::move(committed)))
                            break;
                    } else if (type == "input_audio_buffer.clear" || type == "response.cancel") {
                        stream.reset();
                        partial_transcript.clear();
                        audio_bytes = 0;
                        Value cleared(Value::Object{});
                        cleared["type"] = "input_audio_buffer.cleared";
                        if (!send(std::move(cleared)))
                            break;
                    } else {
                        throw std::invalid_argument("unsupported realtime event type: " + type);
                    }
                }
                catch (const std::exception& error) {
                    Value response(Value::Object{});
                    response["type"] = "error";
                    Value details(Value::Object{});
                    details["message"] = error.what();
                    details["type"] = "invalid_request_error";
                    response["error"] = std::move(details);
                    if (!send(std::move(response)))
                        break;
                }
            }
        };
        server->WebSocket("/v1/realtime", realtime);
        server->WebSocket("/v1/audio/transcriptions/realtime", std::move(realtime));
#endif
    }
};

Server::Server(EngineRegistry& engines, ServerConfig config)
    : impl_(std::make_unique<Impl>(engines, std::move(config))) {}

Server::~Server() = default;

bool
Server::listen() {
    return impl_->server->listen(impl_->config.address, impl_->config.port);
}

bool
Server::bind() {
    return impl_->server->bind_to_port(impl_->config.address, impl_->config.port);
}

bool
Server::listen_after_bind() {
    return impl_->server->listen_after_bind();
}

void
Server::stop() {
    impl_->server->stop();
}

}  // namespace nemo_speech::http
