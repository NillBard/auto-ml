import { IDataset } from "@/shared/types"
import { instance } from "@/shared/libs"

export function getDatasets() {
    return instance.get<IDataset[]>('/api/dataset', { withCredentials: true })
}

export function uploadDataset(id: number, file: File) {
    return instance.post(
        `/api/train/${id}/dataset`,
        { dataset: file },
        {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
            withCredentials: true,
        }
    )
}
