// Enhanced Toolkit v2.0 - Video Processing Component
// Provides utilities for uploading videos, metadata extraction, and transcription

const VideoAnalysis = {
    validateVideoFile(file) {
        const validExtensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];
        const maxSize = 500 * 1024 * 1024; // 500MB
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!validExtensions.includes(ext)) {
            return { valid: false, error: 'Unsupported video format' };
        }
        if (file.size > maxSize) {
            return { valid: false, error: 'Video file too large (500MB limit)' };
        }
        return { valid: true };
    },

    async generatePreview(file) {
        return FileHandler.createFilePreview(file, 'video');
    },

    async extractMetadata(base64Data, filename) {
        try {
            const result = await eel.extract_video_metadata(base64Data, filename)();
            return result;
        } catch (e) {
            console.error('Metadata extraction failed', e);
            return { success: false, error: e.message || 'Metadata error' };
        }
    },

    async transcribeVideo(base64Data, filename) {
        try {
            const result = await eel.transcribe_video_async_web(base64Data, filename)();
            return result;
        } catch (e) {
            console.error('Transcription failed', e);
            return { success: false, error: e.message || 'Transcription error' };
        }
    }
};

window.VideoAnalysis = VideoAnalysis;
