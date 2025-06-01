import { instance } from '@/shared/libs'

export function getMlModels (taskType: string) {
  return instance.get(`/api/train/models/${taskType}`, { withCredentials: true })
} 