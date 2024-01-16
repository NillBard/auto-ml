import { useCallback, useRef, useState } from 'react'
import {
  Flex,
  VStack,
  IconButton,
  Card,
  Image,
  useDisclosure,
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
} from '@chakra-ui/react'
import {
  addEdge,
  Connection,
  Edge,
  Node,
  ReactFlow,
  useEdgesState,
  useNodesState,
  useReactFlow,
} from 'reactflow'

import camera from '../../assets/icons/MdCameraEnhance.svg'
import file from '../../assets/icons/MdCreateNewFolder.svg'
import model from '../../assets/icons/BiCube.svg'

import 'reactflow/dist/style.css'
import { testConnection } from '../../api/api.ts'

const CreatePipeline = () => {
  const [nodes, setNodes, onNodeChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [value, setValue] = useState('')
  const connectingNodeId = useRef(null)
  const { isOpen, onOpen, onClose } = useDisclosure()
  const { screenToFlowPosition } = useReactFlow()

  const [cameraNodeId, setCameraNodeId] = useState(0)
  const [fileNodeId, setFileNodeId] = useState(0)
  const [taskNodeId, setTaskNodeId] = useState(0)

  const addNode = useCallback(
    (name: string) => {
      const getId = () => {
        switch (name) {
          case 'camera':
            setCameraNodeId(cameraNodeId + 1)
            return `camera ${cameraNodeId}`
          case 'file':
            setFileNodeId(fileNodeId + 1)
            return `file ${fileNodeId}`
          case 'task':
            setTaskNodeId(taskNodeId + 1)
            return `task ${taskNodeId}`
          default:
            return 'null'
        }
      }
      const id = getId()
      const newNode = {
        id,
        position: screenToFlowPosition({
          x: 200,
          y: 200,
        }),
        data: {
          label: `${id}`,
        },
        origin: [0.5, 0.0],
      }

      setNodes((nds) => nds.concat(newNode))
    },
    [cameraNodeId, fileNodeId, screenToFlowPosition, setNodes, taskNodeId]
  )

  const onConnect = useCallback(
    (params: Edge | Connection) => {
      connectingNodeId.current = null
      setEdges((eds) => addEdge(params, eds))
    },
    [setEdges]
  )

  const ButtonAction = () => {
    testConnection(value)
      .then((response) => console.log(response))
      .catch((e) => console.log(e))
  }

  const onNodeClick = useCallback(
    (_, { id }: Node) => {
      if (id.includes('camera')) {
        onOpen()
      } else if (id.includes('file')) {
        console.log('file')
      } else if (id.includes('task')) {
        console.log('task')
      } else {
        console.log('null')
      }
    },
    [onOpen]
  )

  return (
    <Flex w="100%">
      <Flex w="95%" h="100vh" pt="25px" pl="25px" pb="25px">
        <Card w="100%" h="100%" border="1px" borderColor="gray.200">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodeChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeDoubleClick={onNodeClick}
          />
        </Card>
        <Modal isOpen={isOpen} onClose={onClose}>
          <ModalOverlay />
          <ModalContent>
            <ModalHeader>Параметры</ModalHeader>
            <ModalCloseButton />
            <ModalBody>
              <Text>Источник</Text>
              <Input
                placeholder="Адрес rtsp-потока"
                onChange={(e) => {
                  setValue(e.target.value)
                }}
              />
            </ModalBody>

            <ModalFooter>
              <Button mr="10px" onClick={ButtonAction}>
                Тест
              </Button>
              <Button colorScheme="blue" mr={3} onClick={onClose}>
                Close
              </Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
      </Flex>
      <VStack pr="10px" pt="25px" width="5%">
        <IconButton aria-label="camera" onClick={() => addNode('camera')}>
          <Image src={camera} />
        </IconButton>
        <IconButton aria-label="file" onClick={() => addNode('file')}>
          <Image src={file} />
        </IconButton>
        <IconButton aria-label="task" onClick={() => addNode('task')}>
          <Image src={model} />
        </IconButton>
      </VStack>
    </Flex>
  )
}

export default CreatePipeline
