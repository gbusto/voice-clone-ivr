'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

type Voice = { id: number; name: string; voice_id: string };

export default function ProfilePage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [voices, setVoices] = useState<Voice[]>([]);
  const [selectedVoiceId, setSelectedVoiceId] = useState('');
  const [text, setText] = useState('');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [polling, setPolling] = useState(false);
  const hasVoices = voices.length > 0;

  const fetchMe = async () => {
    const r = await fetch('/api/me');
    if (!r.ok) {
      router.push('/login');
      return;
    }
    const j = await r.json();
    setEmail(j.email);
    setPhone(j.phone || '');
    setVoices(j.voices || []);
    if (!selectedVoiceId && j.voices?.[0]) setSelectedVoiceId(j.voices[0].voice_id);
  };

  useEffect(() => { fetchMe(); }, []);

  useEffect(() => {
    if (!polling) return;
    const id = setInterval(fetchMe, 6000);
    return () => clearInterval(id);
  }, [polling]);

  const savePhone = async () => {
    await fetch('/api/me/phone', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ phone }) });
    await fetchMe();
  };

  const createVoice = async () => {
    const r = await fetch('/api/voice/sessions', { method: 'POST' });
    if (r.ok) setPolling(true);
  };

  const speak = async () => {
    if (!selectedVoiceId || !text.trim()) return;
    const r = await fetch('/api/tts', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ voice_id: selectedVoiceId, text }) });
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    setAudioUrl(url);
  };

  const logout = async () => {
    await fetch('/auth/logout', { method: 'POST' });
    router.push('/login');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black py-8 px-4">
      <div className="max-w-3xl mx-auto">
        <div className="bg-gray-800 rounded-2xl shadow-2xl p-8 border border-gray-700">
          {/* Header */}
          <div className="flex justify-between items-center mb-8 pb-6 border-b border-gray-700">
            <div>
              <h2 className="text-3xl font-bold text-white">Your Profile</h2>
              <p className="text-gray-400 mt-1">{email}</p>
            </div>
            <button
              onClick={logout}
              className="px-4 py-2 text-gray-300 hover:text-white font-medium border border-gray-600 rounded-lg hover:bg-gray-700 transition"
            >
              Logout
            </button>
          </div>

          {/* Phone Section */}
          <div className="mb-8">
            <label className="block text-sm font-medium text-gray-300 mb-2">Phone Number</label>
            <div className="flex gap-3">
              <input
                type="tel"
                placeholder="+1 (555) 123-4567"
                value={phone}
                onChange={e => setPhone(e.target.value)}
                className="flex-1 px-4 py-3 bg-gray-700 border border-gray-600 text-white placeholder-gray-400 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition"
              />
              <button
                onClick={savePhone}
                className="px-6 py-3 bg-gray-600 hover:bg-gray-500 text-white font-medium rounded-lg transition"
              >
                Save
              </button>
            </div>
          </div>

          {/* Voice Clone Section */}
          <div className="mb-8 p-6 bg-gray-900/50 rounded-xl border border-indigo-900/50">
            <h3 className="text-lg font-semibold text-white mb-4">Voice Cloning</h3>
            <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
              <div className="flex-1 w-full">
                <label className="block text-sm font-medium text-gray-300 mb-2">Select Voice</label>
                <select
                  value={selectedVoiceId}
                  onChange={e => setSelectedVoiceId(e.target.value)}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 text-white rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition"
                >
                  {hasVoices ? (
                    voices.map(v => <option key={v.voice_id} value={v.voice_id}>{v.name}</option>)
                  ) : (
                    <option value="">No voices yet—create one</option>
                  )}
                </select>
              </div>
              <button
                onClick={createVoice}
                className="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg transition shadow-md hover:shadow-lg whitespace-nowrap mt-auto"
              >
                + Create New Voice
              </button>
            </div>
            {polling && (
              <p className="mt-3 text-sm text-indigo-400 font-medium">
                ⏳ Processing your voice... This may take a minute.
              </p>
            )}
          </div>

          {/* Text to Speech Section */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Text to Speech</h3>
            <textarea
              rows={5}
              placeholder="Type or paste text here to synthesize with your cloned voice..."
              value={text}
              onChange={e => setText(e.target.value)}
              className="w-full px-4 py-3 bg-gray-700 border border-gray-600 text-white placeholder-gray-400 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition resize-none"
            />
            <button
              onClick={speak}
              disabled={!hasVoices || !text.trim()}
              className="mt-4 w-full sm:w-auto px-8 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition shadow-md hover:shadow-lg"
            >
              🎤 Speak
            </button>
            
            {audioUrl && (
              <div className="mt-6 p-4 bg-gray-900/50 rounded-lg border border-gray-700">
                <p className="text-sm font-medium text-gray-300 mb-2">Generated Audio:</p>
                <audio controls src={audioUrl} className="w-full" />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}


