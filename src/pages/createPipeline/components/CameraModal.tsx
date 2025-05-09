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
  HStack,
  Flex,
  Image,
} from '@chakra-ui/react'

interface CameraModalProps {
  isOpen: boolean
  onClose: () => void
  cameraSource: string
  onTestConnection: (url: string) => void
  inputValue: string
  setInputValue: (value: string) => void
}

export const CameraModal = ({
  isOpen,
  onClose,
  cameraSource,
  onTestConnection,
  inputValue,
  setInputValue,
}: CameraModalProps) => {
  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Параметры</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <Text>Источник</Text>
          <HStack>
            <Input
            value={inputValue}
              placeholder="Адрес rtsp-потока"
              onChange={(e) => {
                setInputValue(e.target.value)
              }}
            />
            <Button mr="10px" onClick={() => onTestConnection(inputValue)}>
              Тест
            </Button>
          </HStack>
          <Flex h="150px">
            <Image src={cameraSource} />
          </Flex>
        </ModalBody>

        <ModalFooter>
          <Button colorScheme="blue" mr={3} onClick={onClose}>
            Close
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  )
}

export default CameraModal 