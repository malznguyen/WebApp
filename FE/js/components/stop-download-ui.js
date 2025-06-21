const StopDownloadUI = {
    stopTask(taskId) {
        if (!taskId || typeof eel === 'undefined') return;
        eel.stop_task(taskId)().then(res => {
            if (!res.success) {
                console.error('Stop failed', res.error);
            }
        }).catch(err => console.error(err));
    },

    downloadImageAnalysis(result, filename) {
        if (typeof eel === 'undefined') return;
        eel.download_image_analysis(result, filename)().then(res => {
            if (res.success) {
                const link = document.createElement('a');
                link.href = `temp/${res.path.split('/').pop()}`;
                link.download = filename + '.zip';
                link.click();
            } else {
                console.error('Download failed', res.error);
            }
        });
    }
};

window.StopDownloadUI = StopDownloadUI;
