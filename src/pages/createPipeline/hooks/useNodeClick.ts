import { useCallback, useState } from 'react'
import { Node } from 'reactflow'
import { useDisclosure } from '@chakra-ui/react'

type NodeType = 'photo' | 'video' | 'stream' | 'file' | 'task' | 'model' | 'preview' | 'save'

interface NodeModalState {
  isOpen: boolean
  type: NodeType | null
  inputValue: string
}

export const useNodeClick = () => {
  const [modalState, setModalState] = useState<NodeModalState>({
    isOpen: false,
    type: null,
    inputValue: '',
  })

  const onNodeClick = useCallback(
    (_: React.MouseEvent, { id }: Node) => {
      let type: NodeType | null = null

      if (id.includes('Фото')) type = 'photo'
      else if (id.includes('Видео')) type = 'video'
      else if (id.includes('Поток')) type = 'stream'
      else if (id.includes('Файл')) type = 'file'
      else if (id.includes('Задача')) type = 'task'
      else if (id.includes('Модель')) type = 'model'
      else if (id.includes('Предпросмотр')) type = 'preview'
      else if (id.includes('Сохранение')) type = 'save'

      if (type) {
        setModalState({
          isOpen: true,
          type,
          inputValue: 'rtsp://rtspstream:jyXsWBiiDs2ymGh8nQL-4@zephyr.rtsp.stream/people',
        })
      }
    },
    []
  )

  const onClose = useCallback(() => {
    setModalState((prev) => ({
      ...prev,
      isOpen: false,
      type: null,
      inputValue: '',
    }))
  }, [])

  const setInputValue = useCallback((value: string) => {
    setModalState((prev) => ({
      ...prev,
      inputValue: value,
    }))
  }, [])

  const onSave = useCallback(() => {
    // Здесь можно добавить логику сохранения данных
    console.log('Saving data for node type:', modalState.type, 'with value:', modalState.inputValue)
    onClose()
  }, [modalState.type, modalState.inputValue, onClose])

  return {
    modalState,
    onNodeClick,
    onClose,
    setInputValue,
    onSave,
  }
} 