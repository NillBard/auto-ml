import { useCallback, useRef, useState } from 'react'
import {
  Flex,
  Card,
  useDisclosure,
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
import { checkRTSP, startCam, stopCam } from '../../api/api'
import 'reactflow/dist/style.css'
import { NodePanel, NodeModal, CustomNode } from './components'
import { useNodeClick, useAddNode } from './hooks'
import ReactPlayer from 'react-player'

interface ICameraInfo {
  location: string
  login: string
  password: string
}

const nodeTypes = {
  custom: CustomNode,
}

const getModalTitle = (type: string | null): string => {
  switch (type) {
    case 'photo':
      return 'Настройки фото'
    case 'video':
      return 'Настройки видео'
    case 'stream':
      return 'Настройки потока'
    case 'file':
      return 'Настройки файла'
    case 'task':
      return 'Настройки задачи'
    case 'model':
      return 'Настройки модели'
    case 'preview':
      return 'Настройки предпросмотра'
    case 'save':
      return 'Настройки сохранения'
    default:
      return 'Настройки'
  }
}

const CreatePipeline = () => {
  const [nodes, setNodes, onNodeChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const connectingNodeId = useRef(null)
  const [cameraSource, setCameraSource] = useState('')
  const [streamId, setStreamId] = useState('')

  const { modalState, onNodeClick, onClose: onNodeModalClose, setInputValue: setNodeInputValue, onSave } = useNodeClick()
  const { addNode: createNode } = useAddNode()

  const onCloseAction = () => {
    stopCam(streamId)
    onNodeModalClose()
  }

  const playerRef = useRef<ReactPlayer>(null)

  const addNode = useCallback(
    (name: 'photo' | 'video' | 'stream' | 'file' | 'task' | 'model' | 'preview' | 'save') => {
      const newNode = createNode(name)
      setNodes((nds) => nds.concat(newNode))
    },
    [createNode, setNodes]
  )

  const onConnect = useCallback(
    (params: Edge | Connection) => {
      connectingNodeId.current = null
      setEdges((eds) => addEdge(params, eds))
    },
    [setEdges]
  )
console.log(edges);

  const handleTestConnection = (url: string) => {
    checkRTSP(url)
      .then((response) => {
        console.log(response)
        setStreamId(response.data.stream_id)
        setCameraSource(`api/static/streams/${response.data.stream_id}/hls/playlist.m3u8`)
      })
      .catch((e) => console.log(e))
  }

  return (
    <Flex w="100%">
      <NodePanel onAddNode={addNode} />

      <Flex w="95%" h="100vh" pt="25px" pr="25px" pb="25px">
        <Card w="100%" h="100%" border="1px" borderColor="gray.200">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodeChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeDoubleClick={onNodeClick}
            nodeTypes={nodeTypes}
            fitView
          />
        </Card>

        <NodeModal
          isOpen={modalState.isOpen}
          onClose={onCloseAction}
          title={getModalTitle(modalState.type)}
          inputValue={modalState.inputValue}
          setInputValue={setNodeInputValue}
          onSave={onSave}
          type={modalState.type}
          streamId={streamId}
          cameraSource={cameraSource}
          onTestConnection={handleTestConnection}
        />
      </Flex>
    </Flex>
  )
}

export default CreatePipeline
