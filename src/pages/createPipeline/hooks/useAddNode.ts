import { useCallback, useState } from 'react'
import { Node } from 'reactflow'

type NodeType = 'photo' | 'video' | 'stream' | 'file' | 'task' | 'model' | 'preview' | 'save'

interface NodeConfig {
  prefix: string
  counter: number
  setCounter: (value: number) => void
}

const NODE_TYPES: Record<NodeType, NodeConfig> = {
  photo: { prefix: 'Фото', counter: 0, setCounter: () => {} },
  video: { prefix: 'Видео', counter: 0, setCounter: () => {} },
  stream: { prefix: 'Поток', counter: 0, setCounter: () => {} },
  file: { prefix: 'Файл', counter: 0, setCounter: () => {} },
  task: { prefix: 'Задача', counter: 0, setCounter: () => {} },
  model: { prefix: 'Модель', counter: 0, setCounter: () => {} },
  preview: { prefix: 'Предпросмотр', counter: 0, setCounter: () => {} },
  save: { prefix: 'Сохранение', counter: 0, setCounter: () => {} },
}

export const useAddNode = () => {
  const [photoCounter, setPhotoCounter] = useState(0)
  const [videoCounter, setVideoCounter] = useState(0)
  const [streamCounter, setStreamCounter] = useState(0)
  const [fileCounter, setFileCounter] = useState(0)
  const [taskCounter, setTaskCounter] = useState(0)
  const [modelCounter, setModelCounter] = useState(0)
  const [previewCounter, setPreviewCounter] = useState(0)
  const [saveCounter, setSaveCounter] = useState(0)

  const nodeCounters = {
    photo: { counter: photoCounter, setCounter: setPhotoCounter },
    video: { counter: videoCounter, setCounter: setVideoCounter },
    stream: { counter: streamCounter, setCounter: setStreamCounter },
    file: { counter: fileCounter, setCounter: setFileCounter },
    task: { counter: taskCounter, setCounter: setTaskCounter },
    model: { counter: modelCounter, setCounter: setModelCounter },
    preview: { counter: previewCounter, setCounter: setPreviewCounter },
    save: { counter: saveCounter, setCounter: setSaveCounter },
  }

  const addNode = useCallback(
    (name: NodeType) => {
      const config = NODE_TYPES[name]
      const counter = nodeCounters[name].counter
      const setCounter = nodeCounters[name].setCounter

      const newNode: Node = {
        id: `${config.prefix} ${counter + 1}`,
        type: 'custom',
        position: { x: 100, y: 100 },
        data: { label: `${config.prefix} ${counter + 1}` },
      }

      setCounter(counter + 1)
      return newNode
    },
    [nodeCounters]
  )

  return { addNode }
} 