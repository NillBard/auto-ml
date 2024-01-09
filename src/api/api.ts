import axios from './axios.ts'

export async function testConnection(source: string) {
  return await axios.post('/pipeline/test', {
    source: source,
  })
}
