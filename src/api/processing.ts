import instance from './axios.ts'

export function getAllProcessings() {
    return instance.get(
        '/api/processing/processing',
    )
}

export function getProcessing(id: string) {
    return instance.get(
        `/api/processing/processing/${id}`,
    )
}

export interface IProcessingCreate {
    type: string
    rtsp_url: string
    model: string
}
export interface IProcessing {
    id: number
    status: string
    type: "rtsp"
    rtsp_url: string
    created_at: string
}

export function createProcessing(processing: IProcessingCreate) {
    return instance.post(
        '/api/processing/processing',
        processing,
    )
}

export function checkPipeline(rtspURL: string, model: string) {
    return instance.post(
        `/api/processing/start-stream`,
        {
            rtsp_url: rtspURL,
        }
    )
}

export function stopCheckPipeline(streamID: string) {
    return instance.post(
        `/api/processing/stop-stream/${streamID}`
    )
}
