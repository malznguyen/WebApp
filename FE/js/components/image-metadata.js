// Image Metadata Component
// Provides utilities to format and export metadata

const ImageMetadata = {
    formatForDisplay(result) {
        if (!result || !result.success) {
            return null;
        }
        const basic = result.basic || {};
        const exif = result.exif || {};
        const fingerprints = result.fingerprints || {};
        return {
            basic,
            exif,
            fingerprints,
            warnings: result.warnings || []
        };
    },

    exportMetadata(result, filename = 'metadata', format = 'json') {
        if (!result || !result.success) return;
        const ts = new Date().toISOString().split('T')[0];
        if (format === 'json') {
            const data = JSON.stringify(result, null, 2);
            Common.downloadAsFile(data, `${filename}_${ts}.json`, 'application/json');
        } else if (format === 'csv') {
            const rows = [];
            const pushRow = (k, v) => { rows.push(`"${k}","${String(v).replace(/"/g,'""')}"`); };
            Object.entries(result.basic || {}).forEach(([k,v])=>pushRow(k,v));
            Object.entries(result.exif || {}).forEach(([k,v])=>pushRow(k,v));
            Object.entries(result.fingerprints || {}).forEach(([k,v])=>pushRow(k,v));
            const csv = rows.join('\n');
            Common.downloadAsFile(csv, `${filename}_${ts}.csv`, 'text/csv');
        }
    }
};

window.ImageMetadata = ImageMetadata;
