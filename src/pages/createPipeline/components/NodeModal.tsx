import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalCloseButton,
  ModalBody,
  ModalFooter,
  Button,
  Text,
  Input,
  VStack,
  Box,
} from '@chakra-ui/react'
import ReactPlayer from 'react-player'
import { useState } from 'react'
import { StreamViewer } from '../../../components/StreamViewer';


interface CameraInfo {
  location: string
  login: string
  password: string
}

interface NodeModalProps {
  isOpen: boolean
  onClose: (cameraInfo?: CameraInfo) => void
  title: string
  inputValue: string
  setInputValue: (value: string) => void
  onSave: () => void
  type: 'photo' | 'video' | 'stream' | 'file' | 'task' | 'model' | 'preview' | 'save' | null
  cameraSource?: string
  onTestConnection?: (url: string, login: string, password: string) => void
}

export const NodeModal = ({
  isOpen,
  onClose,
  title,
  inputValue,
  setInputValue,
  onSave,
  type,
  cameraSource = '',
  onTestConnection,
  streamId,
}: NodeModalProps) => {
  const isStreamType = type === 'stream'
  const [login, setLogin] = useState('')
  const [password, setPassword] = useState('')

  const handleClose = () => {
    if (isStreamType) {
      onClose({
        location: inputValue,
        login,
        password,
      })
    } else {
      onClose()
    }
  }
  const [flag, setFlag] = useState(0);

  const handleRefresh = () => {
    setFlag(Date.now())
  }
  return (
    <Modal isOpen={isOpen} onClose={handleClose}>
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>{title}</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          {isStreamType ? (
            <VStack spacing={4}>
              <VStack width="100%" spacing={2}>
                <Text width="100%">Адрес RTSP потока</Text>
                <Input
                  placeholder="rtsp://имя_пользователя:пароль@адрес_потока"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                />
              </VStack>
              <Button
                width="100%"
                onClick={() => onTestConnection?.(inputValue, login, password)}
              >
                Тест подключения
              </Button>
            </VStack>
          ) : (
            <VStack spacing={4}>
              <Text>Название</Text>
              <Input
                placeholder="Введите название"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
              />
            </VStack>
          )}
        </ModalBody>

        <ModalFooter>
          <Button colorScheme="blue" mr={3} onClick={onSave}>
            Сохранить
          </Button>
          <Button variant="ghost" onClick={handleClose}>
            Отмена
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  )
} 