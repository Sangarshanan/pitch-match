import streamlit as st
import numpy as np
import pyworld as pw
from swift_f0 import SwiftF0, segment_notes
from st_audiorec import st_audiorec
import time
import io

# Configuration
SAMPLE_RATE = 44100
DURATION = 2
CHANNELS = 1
NOTES = ["C", "D", "E", "F", "G", "A", "B"]

# Initialize the pitch detector
@st.cache_resource
def get_detector():
    return SwiftF0(fmin=46.875, fmax=2093.75, confidence_threshold=0.9)

def pitch_detect_from_file(file_path, detector):
    """Detect pitch from audio file (supports both local files and URLs)"""
    if file_path.startswith('http'):
        # For URLs, load audio data first then use detect_from_array
        audio_data = load_audio_file(file_path)
        result = detector.detect_from_array(audio_data, sample_rate=SAMPLE_RATE)
    else:
        # For local files, use detect_from_file directly
        result = detector.detect_from_file(file_path)

    source_notes = segment_notes(
        result,
        split_semitone_threshold=0.8,
        min_note_duration=0.3
    )
    return source_notes

def pitch_detect_from_array(audio_data, detector, sample_rate=SAMPLE_RATE):
    """Detect pitch from audio array - fail hard"""
    result = detector.detect_from_array(audio_data, sample_rate=sample_rate)
    notes = segment_notes(
        result,
        split_semitone_threshold=0.8,
        min_note_duration=0.3
    )
    # Print debug info before returning
    print(f"Audio shape: {audio_data.shape}")
    print(f"Result type: {type(result)}")
    if len(notes) > 0:
        print(f"First note attributes: {dir(notes[0])}")
    return notes

def calculate_pitch_accuracy(recorded_segments, target_segments):
    """Calculate how accurately the pitch was matched - fail hard"""
    total_error = 0

    for rec_seg, tgt_seg in zip(recorded_segments, target_segments):
        error = abs(tgt_seg.pitch_midi - rec_seg.pitch_midi)
        total_error += error

    avg_error = total_error / len(recorded_segments)
    accuracy = max(0, 100 - (avg_error / 12 * 100))
    return accuracy


def midi_to_note_name(midi_number):
    """Convert MIDI number to note name"""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = int(midi_number // 12) - 1
    note = note_names[int(midi_number % 12)]
    return f"{note}{octave}"

def show_pitch_analysis(detected_notes):
    """Show simple pitch analysis - note and frequency only"""
    if len(detected_notes) > 0:
        st.subheader("🎵 Your Recording")

    for i, note in enumerate(detected_notes):
        # Convert MIDI to frequency
        freq_hz = 440 * (2 ** ((note.pitch_midi - 69) / 12))
        note_name = midi_to_note_name(note.pitch_midi)

        # Create two columns for note and frequency
        col1, col2 = st.columns(2)

        with col1:
            st.metric("🎼 Note", note_name)
        with col2:
            st.metric("🔊 Frequency", f"{freq_hz:.1f} Hz")

        if i < len(detected_notes) - 1:
            st.write("---")

def load_audio_file(file_path):
    """Load audio file and convert to numpy array (supports URLs)"""
    import librosa
    import urllib.request
    import io

    if file_path.startswith('http'):
        # For URLs, download the file and load from bytes
        with urllib.request.urlopen(file_path) as response:
            audio_bytes = io.BytesIO(response.read())
            audio, sr = librosa.load(audio_bytes, sr=SAMPLE_RATE)
    else:
        # For local files, load directly
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    return audio

def create_overlaid_audio(target_file_path, user_recording):
    """Create overlaid audio by mixing target file with user recording"""
    import librosa

    # Load target audio
    target_audio = load_audio_file(target_file_path)

    # Ensure user recording is 1D
    if len(user_recording.shape) > 1:
        user_audio = user_recording.flatten()
    else:
        user_audio = user_recording.copy()

    # Make both arrays the same length (pad shorter one with zeros)
    max_len = max(len(target_audio), len(user_audio))

    if len(target_audio) < max_len:
        target_audio = np.pad(target_audio, (0, max_len - len(target_audio)))
    if len(user_audio) < max_len:
        user_audio = np.pad(user_audio, (0, max_len - len(user_audio)))

    # Mix the audio (overlay) - reduce volume of each by 50% to prevent clipping
    overlaid = (target_audio * 0.5) + (user_audio * 0.5)

    return overlaid

def create_overlaid_audio_arrays(audio1, audio2):
    """Create overlaid audio by mixing two numpy arrays"""
    # Ensure both are 1D
    if len(audio1.shape) > 1:
        audio1 = audio1.flatten()
    if len(audio2.shape) > 1:
        audio2 = audio2.flatten()

    # Make both arrays the same length (pad shorter one with zeros)
    max_len = max(len(audio1), len(audio2))

    if len(audio1) < max_len:
        audio1 = np.pad(audio1, (0, max_len - len(audio1)))
    if len(audio2) < max_len:
        audio2 = np.pad(audio2, (0, max_len - len(audio2)))

    # Mix the audio (overlay) - reduce volume of each by 50% to prevent clipping
    overlaid = (audio1 * 0.5) + (audio2 * 0.5)

    return overlaid

def pitch_shift_world(audio_data, recorded_segments, target_segments, sample_rate=44100):
    """Apply pitch shifting using WORLD vocoder"""
    if len(audio_data.shape) > 1:
        audio = audio_data.flatten()
    else:
        audio = audio_data.copy()

    audio = audio.astype(np.float64)

    # Calculate average pitch error
    total_shift = 0
    print("Analyzing pitch differences:")
    print("-" * 70)

    for i, (rec_seg, tgt_seg) in enumerate(zip(recorded_segments, target_segments)):
        semitones_shift = tgt_seg.pitch_midi - rec_seg.pitch_midi
        total_shift += semitones_shift

    # Calculate average shift
    avg_semitones = total_shift / len(recorded_segments)
    semitones = round(avg_semitones)

    print("\n" + "="*70)
    print(f"Average shift needed: {avg_semitones:.2f} semitones")
    print(f"Applying pitch shift: {semitones:+d} semitones")

    # Extract features using WORLD vocoder
    f0, sp, ap = pw.wav2world(audio, sample_rate)

    # Calculate pitch shift ratio
    pitch_ratio = 2 ** (semitones / 12.0)

    # Shift the fundamental frequency (F0) for entire audio
    f0_shifted = f0 * pitch_ratio

    # Synthesize audio with shifted pitch
    shifted_audio = pw.synthesize(f0_shifted, sp, ap, sample_rate)

    return shifted_audio.astype(np.float32)

def main():
    st.set_page_config(
        page_title="Pitch Match",
        page_icon="♪",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Initialize session state
    if 'game_state' not in st.session_state:
        st.session_state.game_state = 'select_key'
    if 'selected_note' not in st.session_state:
        st.session_state.selected_note = None
    if 'detector' not in st.session_state:
        st.session_state.detector = get_detector()
    if 'current_round' not in st.session_state:
        st.session_state.current_round = 'piano'
    if 'piano_recording' not in st.session_state:
        st.session_state.piano_recording = None
    if 'vocal_recording' not in st.session_state:
        st.session_state.vocal_recording = None
    if 'corrected_recording' not in st.session_state:
        st.session_state.corrected_recording = None
    if 'target_notes' not in st.session_state:
        st.session_state.target_notes = None
    if 'shifted_audio' not in st.session_state:
        st.session_state.shifted_audio = None

    # Game flow
    if st.session_state.game_state == 'select_key':
        show_key_selection()
    elif st.session_state.game_state == 'play_and_record':
        show_play_and_record()
    elif st.session_state.game_state == 'corrected_round':
        show_corrected_round()
    elif st.session_state.game_state == 'results':
        show_simple_results()

def show_key_selection():
    st.title("♪ Pitch Match")
    st.subheader("Choose a Note")

    st.write("Select a musical note to practice pitch matching:")

    # Create buttons for each note in a grid layout
    cols = st.columns(7)
    for i, note in enumerate(NOTES):
        with cols[i]:
            if st.button(f"**{note}3**", key=f"note_{note}", use_container_width=True):
                st.session_state.selected_note = note
                st.session_state.game_state = 'play_and_record'
                st.session_state.current_round = 'piano'
                st.rerun()

def show_play_and_record():
    note = st.session_state.selected_note
    round_name = st.session_state.current_round

    st.title(f"♪ {note}3 - {round_name.title()}")

    # Determine file path based on current round
    if round_name == 'piano':
        file_path = f"https://github.com/Sangarshanan/pitch-match/raw/refs/heads/main/media/piano/{note.lower()}3.wav"
        instruction = "Listen to the piano note, then try to match it with your voice"
    else:
        file_path = f"https://github.com/Sangarshanan/pitch-match/raw/refs/heads/main/media/voice/{note}3_Bass.mp3"
        instruction = "Listen to the vocal sample, then try to match it"

    st.write(instruction)
    st.write("")

    # Audio play box
    st.subheader("🎵 Play Target Audio")
    st.audio(file_path)

    # Load target notes for comparison
    if st.session_state.target_notes is None:
        st.session_state.target_notes = pitch_detect_from_file(file_path, st.session_state.detector)

    st.write("")
    st.subheader("🎤 Record Your Voice")
    st.write("Click the microphone to start recording, then click STOP when finished.")

    # Use separate audio recorders for each round
    if round_name == 'piano':
        wav_audio_data = st_audiorec()  # Piano recorder
        recording_key = "piano_raw_recording"
        processed_key = "piano_recording"
    else:  # vocal round
        wav_audio_data = st_audiorec()  # Vocal recorder
        recording_key = "vocal_raw_recording"
        processed_key = "vocal_recording"

    if wav_audio_data is not None:
        # Check if audio has meaningful content
        try:
            import librosa
            audio_io = io.BytesIO(wav_audio_data)
            audio_array, sr = librosa.load(audio_io, sr=SAMPLE_RATE, mono=True)

            if len(audio_array) > 0 and np.max(np.abs(audio_array)) > 1e-6:
                # Store the raw recording
                st.session_state[recording_key] = audio_array
        except:
            pass

    # Show Done button if we have a recording but haven't processed it yet
    if (recording_key in st.session_state and
        st.session_state[recording_key] is not None and
        st.session_state.get(processed_key) is None):


        if st.button("✅ Done - Process Recording", type="primary", use_container_width=True):
            try:
                audio_array = st.session_state[recording_key]

                # Process and validate the recording
                detected_notes = pitch_detect_from_array(audio_array, st.session_state.detector, SAMPLE_RATE)

                if len(detected_notes) == 0:
                    st.error("🚫 No clear pitch detected in recording. Please record again and speak/sing louder.")
                    # Clear the raw recording so user can try again
                    del st.session_state[recording_key]
                    st.rerun()
                else:
                    # Store the processed recording
                    st.session_state[processed_key] = audio_array
                    # Show pitch analysis
                    show_pitch_analysis(detected_notes)
                    st.rerun()

            except Exception as e:
                st.error(f"🚫 Error processing audio: {str(e)}")
                # Clear the raw recording so user can try again
                if recording_key in st.session_state:
                    del st.session_state[recording_key]
                st.rerun()

    # Show pitch analysis if recording is processed
    elif st.session_state.get(processed_key) is not None:
        audio_array = st.session_state[processed_key]
        detected_notes = pitch_detect_from_array(audio_array, st.session_state.detector, SAMPLE_RATE)
        show_pitch_analysis(detected_notes)

    # Show next button if recording is successful
    if round_name == 'piano' and st.session_state.get('piano_recording') is not None:
        if st.button("➡️ Next: Vocal Round", type="primary", use_container_width=True):
            st.session_state.current_round = 'vocal'
            st.rerun()
    elif round_name == 'vocal' and st.session_state.get('vocal_recording') is not None:
        if st.button("🔧 Next: Corrected Round", type="primary", use_container_width=True):
            # Generate pitch-corrected audio
            vocal_notes = pitch_detect_from_array(st.session_state.vocal_recording, st.session_state.detector, SAMPLE_RATE)

            # Use vocal recording as base and shift to match target
            st.session_state.shifted_audio = pitch_shift_world(
                st.session_state.vocal_recording,
                vocal_notes,
                st.session_state.target_notes,
                SAMPLE_RATE
            )

            st.session_state.game_state = 'corrected_round'
            st.rerun()

    # Navigation
    st.write("")
    if st.button("← Back to Key Selection"):
        # Reset recordings when going back
        st.session_state.piano_recording = None
        st.session_state.vocal_recording = None
        st.session_state.target_notes = None
        # Also clear raw recordings
        if 'piano_raw_recording' in st.session_state:
            del st.session_state['piano_raw_recording']
        if 'vocal_raw_recording' in st.session_state:
            del st.session_state['vocal_raw_recording']
        st.session_state.game_state = 'select_key'
        st.rerun()

def show_corrected_round():
    note = st.session_state.selected_note

    st.title(f"♪ {note}3 - Pitch-Corrected")

    st.write("Listen to your pitch-corrected voice, then try to match it!")

    st.write("")
    st.subheader("🔧 Your Pitch-Corrected Voice")

    if st.session_state.shifted_audio is not None:
        # Play the pitch-corrected audio
        st.audio(st.session_state.shifted_audio, sample_rate=SAMPLE_RATE)

    st.write("")
    st.subheader("🎤 Record Your Voice")
    st.write("Click the microphone to start recording, then click STOP when finished.")

    # Use separate audio recorder for corrected round
    corrected_wav_audio_data = st_audiorec()  # Corrected recorder

    # Store raw recording data in session state
    recording_key = "corrected_raw_recording"
    processed_key = "corrected_recording"

    if corrected_wav_audio_data is not None:
        # Check if audio has meaningful content
        try:
            import librosa
            audio_io = io.BytesIO(corrected_wav_audio_data)
            audio_array, sr = librosa.load(audio_io, sr=SAMPLE_RATE, mono=True)

            if len(audio_array) > 0 and np.max(np.abs(audio_array)) > 1e-6:
                # Store the raw recording
                st.session_state[recording_key] = audio_array
        except:
            pass

    # Show Done button if we have a recording but haven't processed it yet
    if (recording_key in st.session_state and
        st.session_state[recording_key] is not None and
        st.session_state.get(processed_key) is None):


        if st.button("✅ Done - Process Recording", type="primary", use_container_width=True):
            try:
                audio_array = st.session_state[recording_key]

                # Process and validate the recording
                detected_notes = pitch_detect_from_array(audio_array, st.session_state.detector, SAMPLE_RATE)

                if len(detected_notes) == 0:
                    st.error("🚫 No clear pitch detected in recording. Please record again and speak/sing louder.")
                    # Clear the raw recording so user can try again
                    del st.session_state[recording_key]
                    st.rerun()
                else:
                    # Store the processed recording
                    st.session_state[processed_key] = audio_array
                    # Show pitch analysis
                    show_pitch_analysis(detected_notes)
                    st.rerun()

            except Exception as e:
                st.error(f"🚫 Error processing audio: {str(e)}")
                # Clear the raw recording so user can try again
                if recording_key in st.session_state:
                    del st.session_state[recording_key]
                st.rerun()

    # Show pitch analysis if recording is processed
    elif st.session_state.get(processed_key) is not None:
        audio_array = st.session_state[processed_key]
        detected_notes = pitch_detect_from_array(audio_array, st.session_state.detector, SAMPLE_RATE)
        show_pitch_analysis(detected_notes)

    # Show results button if recording is successful
    if st.session_state.get('corrected_recording') is not None:
        if st.button("📊 View Results", type="primary", use_container_width=True):
            st.session_state.game_state = 'results'
            st.rerun()

    # Navigation
    st.write("")
    if st.button("← Back to Key Selection"):
        # Reset recordings when going back
        st.session_state.piano_recording = None
        st.session_state.vocal_recording = None
        st.session_state.corrected_recording = None
        st.session_state.target_notes = None
        st.session_state.shifted_audio = None
        # Also clear raw recordings
        if 'piano_raw_recording' in st.session_state:
            del st.session_state['piano_raw_recording']
        if 'vocal_raw_recording' in st.session_state:
            del st.session_state['vocal_raw_recording']
        if 'corrected_raw_recording' in st.session_state:
            del st.session_state['corrected_raw_recording']
        st.session_state.game_state = 'select_key'
        st.rerun()

def show_simple_results():
    note = st.session_state.selected_note
    st.title(f"📊 Results - {note}3")

    if (st.session_state.piano_recording is not None and
        st.session_state.vocal_recording is not None and
        st.session_state.corrected_recording is not None and
        st.session_state.target_notes is not None):

        # Analyze recordings with fixed sample rate
        piano_notes = pitch_detect_from_array(st.session_state.piano_recording, st.session_state.detector, SAMPLE_RATE)
        vocal_notes = pitch_detect_from_array(st.session_state.vocal_recording, st.session_state.detector, SAMPLE_RATE)
        corrected_notes = pitch_detect_from_array(st.session_state.corrected_recording, st.session_state.detector, SAMPLE_RATE)

        # Calculate accuracies
        piano_accuracy = calculate_pitch_accuracy(piano_notes, st.session_state.target_notes)
        vocal_accuracy = calculate_pitch_accuracy(vocal_notes, st.session_state.target_notes)

        # For corrected round, compare against the shifted audio instead of target
        shifted_notes = pitch_detect_from_array(st.session_state.shifted_audio, st.session_state.detector, SAMPLE_RATE)
        corrected_accuracy = calculate_pitch_accuracy(corrected_notes, shifted_notes)

        # Show top message based on best performance
        best_accuracy = max(piano_accuracy, vocal_accuracy, corrected_accuracy)
        if piano_accuracy == best_accuracy:
            st.success("🎹 You were best at pitch matching with the piano!")
        elif vocal_accuracy == best_accuracy:
            st.success("🎤 You were best at pitch matching with the vocal sample!")
        else:
            st.success("🔧 You were best at pitch matching with your corrected voice!")

        st.write("---")

        # Piano comparison section
        st.subheader("🎹 Piano Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Target Piano:**")
            st.audio(f"https://github.com/Sangarshanan/pitch-match/raw/refs/heads/main/media/piano/{note.lower()}3.wav")

            st.write("**Your Recording:**")
            st.audio(st.session_state.piano_recording, sample_rate=SAMPLE_RATE)

        with col2:
            st.metric("Piano Match Accuracy", f"{piano_accuracy:.1f}%")
            if piano_notes:
                # Show what you sang vs target
                user_note = midi_to_note_name(piano_notes[0].pitch_midi)
                target_note = midi_to_note_name(st.session_state.target_notes[0].pitch_midi)
                user_freq = 440 * (2 ** ((piano_notes[0].pitch_midi - 69) / 12))
                target_freq = 440 * (2 ** ((st.session_state.target_notes[0].pitch_midi - 69) / 12))
                st.write(f"**You sang:** {user_note} ({user_freq:.1f} Hz)")
                st.write(f"**Target:** {target_note} ({target_freq:.1f} Hz)")

        st.write("---")

        # Vocal comparison section
        st.subheader("🎤 Vocal Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Target Vocal:**")
            st.audio(f"https://github.com/Sangarshanan/pitch-match/raw/refs/heads/main/media/voice/{note}3_Bass.mp3")

            st.write("**Your Recording:**")
            st.audio(st.session_state.vocal_recording, sample_rate=SAMPLE_RATE)

        with col2:
            st.metric("Vocal Match Accuracy", f"{vocal_accuracy:.1f}%")
            if vocal_notes:
                # Show what you sang vs target
                user_note = midi_to_note_name(vocal_notes[0].pitch_midi)
                target_note = midi_to_note_name(st.session_state.target_notes[0].pitch_midi)
                user_freq = 440 * (2 ** ((vocal_notes[0].pitch_midi - 69) / 12))
                target_freq = 440 * (2 ** ((st.session_state.target_notes[0].pitch_midi - 69) / 12))
                st.write(f"**You sang:** {user_note} ({user_freq:.1f} Hz)")
                st.write(f"**Target:** {target_note} ({target_freq:.1f} Hz)")

        st.write("---")

        # Corrected comparison section
        st.subheader("🔧 Corrected Voice Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Target (Pitch-Corrected Voice):**")
            st.audio(st.session_state.shifted_audio, sample_rate=SAMPLE_RATE)

            st.write("**Your Recording:**")
            st.audio(st.session_state.corrected_recording, sample_rate=SAMPLE_RATE)

        with col2:
            st.metric("Corrected Match Accuracy", f"{corrected_accuracy:.1f}%")
            if corrected_notes and shifted_notes:
                # Show what you sang vs corrected target
                user_note = midi_to_note_name(corrected_notes[0].pitch_midi)
                target_note = midi_to_note_name(shifted_notes[0].pitch_midi)
                user_freq = 440 * (2 ** ((corrected_notes[0].pitch_midi - 69) / 12))
                target_freq = 440 * (2 ** ((shifted_notes[0].pitch_midi - 69) / 12))
                st.write(f"**You sang:** {user_note} ({user_freq:.1f} Hz)")
                st.write(f"**Target:** {target_note} ({target_freq:.1f} Hz)")

        st.write("---")

        # Overall performance
        avg_accuracy = (piano_accuracy + vocal_accuracy + corrected_accuracy) / 3
        st.subheader("🏆 Overall Performance")

        if avg_accuracy >= 80:
            st.success(f"Excellent! {avg_accuracy:.1f}% average accuracy across all rounds")
            st.balloons()
        elif avg_accuracy >= 60:
            st.info(f"Good job! {avg_accuracy:.1f}% average accuracy across all rounds")
        else:
            st.warning(f"Keep practicing! {avg_accuracy:.1f}% average accuracy across all rounds")

    else:
        st.error("Missing recordings. Please complete all three rounds.")


    st.write("")

    # Navigation buttons
    if st.button("🔄 Try Again", use_container_width=True):
        # Reset for same note
        st.session_state.piano_recording = None
        st.session_state.vocal_recording = None
        st.session_state.corrected_recording = None
        st.session_state.target_notes = None
        st.session_state.shifted_audio = None
        # Also clear raw recordings
        if 'piano_raw_recording' in st.session_state:
            del st.session_state['piano_raw_recording']
        if 'vocal_raw_recording' in st.session_state:
            del st.session_state['vocal_raw_recording']
        if 'corrected_raw_recording' in st.session_state:
            del st.session_state['corrected_raw_recording']
        st.session_state.current_round = 'piano'
        st.session_state.game_state = 'play_and_record'
        st.rerun()

    if st.button("🎵 Choose New Key", use_container_width=True):
        # Reset everything
        st.session_state.piano_recording = None
        st.session_state.vocal_recording = None
        st.session_state.corrected_recording = None
        st.session_state.target_notes = None
        st.session_state.shifted_audio = None
        st.session_state.selected_note = None
        # Also clear raw recordings
        if 'piano_raw_recording' in st.session_state:
            del st.session_state['piano_raw_recording']
        if 'vocal_raw_recording' in st.session_state:
            del st.session_state['vocal_raw_recording']
        if 'corrected_raw_recording' in st.session_state:
            del st.session_state['corrected_raw_recording']
        st.session_state.current_round = 'piano'
        st.session_state.game_state = 'select_key'
        st.rerun()

if __name__ == "__main__":
    main()