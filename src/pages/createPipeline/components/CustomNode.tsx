import { memo } from 'react'
import { Handle, Position } from 'reactflow'
import { Box, Text, Icon } from '@chakra-ui/react'
import { Icon as Iconify } from '@iconify/react'

type NodeType = 'photo' | 'video' | 'stream' | 'file' | 'task' | 'model' | 'preview' | 'save'

interface CustomNodeProps {
  data: {
    label: string
  }
}

const getNodeIcon = (label: string): string => {
  if (label.includes('Фото')) return 'mdi:image'
  if (label.includes('Видео')) return 'mdi:video'
  if (label.includes('Поток')) return 'mdi:webcam'
  if (label.includes('Файл')) return 'mdi:file-document'
  if (label.includes('Задача')) return 'mdi:cube'
  if (label.includes('Модель')) return 'mdi:brain'
  if (label.includes('Предпросмотр')) return 'mdi:eye'
  if (label.includes('Сохранение')) return 'mdi:content-save'
  return 'mdi:help'
}

const CustomNode = ({ data }: CustomNodeProps) => {
  const icon = getNodeIcon(data.label)

  return (
    <Box
      borderRadius="md"
      bg="white"
      border="1px"
      borderColor="gray.200"
      boxShadow="sm"
      minWidth="100px"
      _hover={{ boxShadow: 'md' }}
    >
      <Handle type="target" position={Position.Top} style={{ background: '#555' }} />
      <Box display="flex" alignItems="center" justifyContent="center" p={1}>
        <Box
          display="flex"
          alignItems="center"
          justifyContent="center"
          width="16px"
          height="16px"
          bg="#3182CE"
          borderRadius="sm"
        >
          <Iconify icon={icon} width="12" height="12" color="white" />
        </Box>
        <Box display="flex" alignItems="center" justifyContent="center" flexGrow={1}>
          <Text fontSize="8px" fontWeight="medium" textAlign="center">
            {data.label}
          </Text>
        </Box>
      </Box>
      <Handle type="source" position={Position.Bottom} style={{ background: '#555' }} />
    </Box>
  )
}

export default memo(CustomNode) 