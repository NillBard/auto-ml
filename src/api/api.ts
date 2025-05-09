import instance from './axios.ts'
import axios, { AxiosResponse } from 'axios'

import { ICameraInfo, ITrain, ITrainCreate, IDataset } from '../types/train.ts'

export function login(email: string, password: string) {
  return axios.post(
    '/api/user',
    {
      login: email,
      password,
    },
    { withCredentials: true }
  )
}

export function logout() {
  return instance.delete('/api/user', { withCredentials: true })
}

export function register(email: string, password: string) {
  return axios.post('/api/user/register', {
    login: email,
    password,
  })
}

export function testConnection(source: string) {
  return instance.post(
    '/api/pipeline/test',
    {
      source: source,
    },
    { withCredentials: true }
  )
}

export function getTrainingConfigurations() {
  return instance.get<ITrain[]>('/api/train/all', { withCredentials: true })
}

export function getDatasets() {
  return instance.get<IDataset[]>('/api/dataset', { withCredentials: true })
}

export function createTrainingConfiguration(configuration: ITrainCreate) {
  return instance.post<ITrain>('/api/train/', configuration, {
    withCredentials: true,
  })
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

export function startCam(cameraInfo: ICameraInfo) {
  return instance.post(
    `/api/pipeline/start`, cameraInfo, {
      withCredentials: true,
    }
  )
}

export function checkRTSP(url: string) {
  return instance.post(
    `/api/processing/start-stream`, {
      rtsp_url: url,
    }
  )
}

export function stopCam(stream_id: string) {
  return instance.post(
    `/api/processing/stop-stream/${stream_id}`
  )
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

export interface RefreshResponse {
  refresh: string
}

export type TPromiseRefresh = Promise<AxiosResponse<RefreshResponse>>

export function refresh(token: string) {
  return axios.post<RefreshResponse>(
    '/api/user/refresh',
    { refresh: token },
    {
      withCredentials: true,
    }
  )
}

// Руками не трогать может взорваться!!!

let refreshPromise: TPromiseRefresh | null

export async function refreshWithoutRepeats() {
  const localCopy = refreshPromise
  let response: AxiosResponse<RefreshResponse>
  if (localCopy && refreshPromise) {
    response = await refreshPromise
  } else {
    refreshPromise = refresh(localStorage.getItem('refresh') || '')
    const copy: TPromiseRefresh = refreshPromise
    response = await copy
    refreshPromise = null
  }

  if (response.data && response.data.refresh) {
    localStorage.setItem('refresh', response.data.refresh)
  } else {
    localStorage.removeItem('refresh')
  }
}
