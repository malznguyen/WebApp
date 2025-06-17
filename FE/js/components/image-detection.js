const ImageDetection = {
    detectionCategories: [
        { min: 80, color: 'var(--red-400)', message: 'R\u1ea5t c\u00f3 kh\u1ea3 n\u0103ng l\u00e0 \u1ea3nh AI' },
        { min: 60, color: 'orange', message: 'C\u00f3 kh\u1ea3 n\u0103ng l\u00e0 \u1ea3nh AI' },
        { min: 40, color: 'var(--yellow-400)', message: 'Kh\u00f4ng ch\u1eafc ch\u1eafn' },
        { min: 20, color: 'lightgreen', message: '\u00cdt kh\u1ea3 n\u0103ng l\u00e0 \u1ea3nh AI' },
        { min: 0,  color: 'var(--green-400)', message: 'R\u1ea5t \u00edt kh\u1ea3 n\u0103ng l\u00e0 \u1ea3nh AI' }
    ],

    getCategory(prob) {
        for (const c of this.detectionCategories) {
            if (prob >= c.min) return c;
        }
        return this.detectionCategories[this.detectionCategories.length - 1];
    },

    probabilityStyle(prob) {
        const cat = this.getCategory(prob);
        return `width: ${prob}% ; background: ${cat.color}`;
    },

    formatDetectionResult(result) {
        if (!result || !result.success) {
            return { error: result?.error || 'Detection failed' };
        }
        const data = result.detection || {};
        return {
            probability: parseFloat(data.ai_generated_probability) || 0,
            confidence: data.confidence_level || '',
            summary: data.analysis_summary || '',
            indicators: Array.isArray(data.detected_indicators) ? data.detected_indicators : [],
            method: data.likely_generation_method || 'Unknown',
            raw: result.raw_response
        };
    }
};

window.ImageDetection = ImageDetection;
