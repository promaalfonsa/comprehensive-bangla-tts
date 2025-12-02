"""
Flask Recording App for Bangla TTS Data Collection

This app provides a web interface for recording audio samples
from sentences defined in the training CSV file.
"""

import os
import csv
import wave
import struct
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# Configuration
SAMPLE_RATE = 22050
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'training sentence data.csv')
WAVS_DIR = os.path.join(BASE_DIR, 'data', 'wavs')

# Ensure wavs directory exists
os.makedirs(WAVS_DIR, exist_ok=True)


def load_sentences():
    """Load sentences from the CSV file."""
    sentences = []
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sentences.append({
                    'id': row['Sentence #'],
                    'text': row['Standard_Bangla']
                })
    except FileNotFoundError:
        raise FileNotFoundError(f"Training data CSV not found at: {CSV_PATH}")
    except KeyError as e:
        raise ValueError(f"CSV file missing required column: {e}")
    return sentences


def get_audio_path(sentence_id):
    """Get the audio file path for a sentence ID."""
    return os.path.join(WAVS_DIR, f'{sentence_id}.wav')


def audio_exists(sentence_id):
    """Check if audio file exists for a sentence."""
    return os.path.exists(get_audio_path(sentence_id))


@app.route('/')
def index():
    """Main recording page."""
    sentences = load_sentences()
    return render_template('index.html', 
                          sentences=sentences,
                          total=len(sentences),
                          sample_rate=SAMPLE_RATE)


@app.route('/api/sentences')
def get_sentences():
    """API endpoint to get all sentences."""
    sentences = load_sentences()
    # Add recording status to each sentence
    for sentence in sentences:
        sentence['recorded'] = audio_exists(sentence['id'])
    return jsonify(sentences)


@app.route('/api/sentence/<sentence_id>')
def get_sentence(sentence_id):
    """API endpoint to get a specific sentence."""
    sentences = load_sentences()
    for sentence in sentences:
        if sentence['id'] == sentence_id:
            sentence['recorded'] = audio_exists(sentence['id'])
            return jsonify(sentence)
    return jsonify({'error': 'Sentence not found'}), 404


@app.route('/api/save', methods=['POST'])
def save_audio():
    """Save recorded audio as 16-bit PCM WAV."""
    try:
        data = request.get_json()
        sentence_id = data.get('sentence_id')
        audio_data = data.get('audio_data')  # List of float samples (-1 to 1)
        
        if not sentence_id or not audio_data:
            return jsonify({'error': 'Missing sentence_id or audio_data'}), 400
        
        # Convert float samples to 16-bit PCM
        audio_path = get_audio_path(sentence_id)
        
        with wave.open(audio_path, 'w') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH)
            wav_file.setframerate(SAMPLE_RATE)
            
            # Convert float [-1, 1] to 16-bit signed integers efficiently
            packed_data = b''.join(
                struct.pack('<h', int(max(-1.0, min(1.0, sample)) * 32767))
                for sample in audio_data
            )
            wav_file.writeframes(packed_data)
        
        return jsonify({
            'success': True,
            'message': f'Audio saved for sentence {sentence_id}',
            'path': audio_path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete/<sentence_id>', methods=['DELETE'])
def delete_audio(sentence_id):
    """Delete recorded audio for a sentence."""
    audio_path = get_audio_path(sentence_id)
    if os.path.exists(audio_path):
        os.remove(audio_path)
        return jsonify({
            'success': True,
            'message': f'Audio deleted for sentence {sentence_id}'
        })
    return jsonify({'error': 'Audio file not found'}), 404


@app.route('/api/audio/<sentence_id>')
def get_audio(sentence_id):
    """Get audio file for a sentence."""
    audio_path = get_audio_path(sentence_id)
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype='audio/wav')
    return jsonify({'error': 'Audio file not found'}), 404


@app.route('/api/status')
def get_status():
    """Get recording status summary."""
    sentences = load_sentences()
    recorded_count = sum(1 for s in sentences if audio_exists(s['id']))
    return jsonify({
        'total': len(sentences),
        'recorded': recorded_count,
        'remaining': len(sentences) - recorded_count
    })


if __name__ == '__main__':
    # Debug mode can be enabled via FLASK_DEBUG environment variable
    # Default to False for security in production
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
