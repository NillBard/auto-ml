import { useCallback, useRef } from 'react'
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
} from '@chakra-ui/react'
import {
  addEdge,
  Connection,
  Edge,
  ReactFlow,
  useEdgesState,
  useNodesState,
  useReactFlow,
} from 'reactflow'

import camera from '../../assets/icons/MdCameraEnhance.svg'
import file from '../../assets/icons/MdCreateNewFolder.svg'
import model from '../../assets/icons/BiCube.svg'

import 'reactflow/dist/style.css'

let cameraId = 0
let taskId = 0
let fileId = 0
let id = 0

const getId = () => `${id++}`
const getCameraId = () => `${cameraId++}`
const getFileId = () => `${fileId++}`
const getTaskId = () => `${taskId++}`

const CreatePipeline = () => {
  const [nodes, setNodes, onNodeChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const connectingNodeId = useRef(null)
  const { isOpen, onOpen, onClose } = useDisclosure()
  const { screenToFlowPosition } = useReactFlow()

  const addNode = useCallback(
    (name: string) => {
      let nodeId
      const id = getId()

      switch (name) {
        case 'camera':
          nodeId = getCameraId()
          break
        case 'file':
          nodeId = getFileId()
          break
        case 'task':
          nodeId = getTaskId()
          break
      }

      const newNode = {
        id,
        position: screenToFlowPosition({
          x: 100,
          y: 100,
        }),
        data: {
          label: `${name} ${nodeId}`,
        },
        origin: [0.5, 0.0],
      }

      setNodes((nds) => nds.concat(newNode))
    },
    [screenToFlowPosition, setNodes]
  )

  const onConnect = useCallback(
    (params: Edge | Connection) => {
      // reset the start node on connections
      connectingNodeId.current = null
      setEdges((eds) => addEdge(params, eds))
    },
    [setEdges]
  )

  return (
    <Flex>
      <VStack pl="10px" pt="25px">
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

      <Flex w="100vw" h="100vh" pt="25px" pl="25px">
        <Card w="98%" h="90%">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodeChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeDoubleClick={onOpen}
          />
        </Card>
        <Modal isOpen={isOpen} onClose={onClose}>
          <ModalOverlay />
          <ModalContent>
            <ModalHeader>Параметры</ModalHeader>
            <ModalCloseButton />
            <ModalBody></ModalBody>

            <ModalFooter>
              <Button colorScheme="blue" mr={3} onClick={onClose}>
                Close
              </Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
      </Flex>
    </Flex>
  )
}

export default CreatePipeline
