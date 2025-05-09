import { instance } from '../libs/axios.ts';
import { ITrain, ITrainCreate } from '@/shared/types';

export function getTrainingConfigurations() {
    return instance.get<ITrain[]>('/api/train/all', { withCredentials: true })
}

export function createTrainingConfiguration(configuration: ITrainCreate) {
    return instance.post<ITrain>('/api/train/', configuration, {
        withCredentials: true,
    })
}

export function startLearning(id: number) {
    return instance.post(`/api/train/${id}/start`, { withCredentials: true })
}

export function getTrainingResults(id: string) {
    return instance.get<ITrain>(`/api/train/${id}`, { withCredentials: true })
}

export function getFile(id: string, type: string) {
    return instance.get(`/api/train/${id}/${type}`, { withCredentials: true, responseType: 'blob' })
}