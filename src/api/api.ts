import axios from './axios.ts'

import { ITrain, ITrainCreate } from '../types/train.ts'

export function testConnection(source: string) {
  return axios.post('/pipeline/test', {
    source: source,
  })
}


export function getTrainingConfigurations() {
  return axios.get<ITrain[]>('/train/all')
}

export function createTrainingConfiguration(configuration: ITrainCreate) {
  return axios.post<ITrain>('/train/', configuration)
}

export function uploadDataset(id: number, file: File) {
  return axios.post(
    `/train/${id}/dataset`,
    { dataset: file },
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  )
}

export function startLearning(id: number) {
  return axios.post(`/train/${id}/start`)
}
