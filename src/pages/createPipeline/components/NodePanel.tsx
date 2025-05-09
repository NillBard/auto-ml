import { List, ListItem, IconButton, Text, VStack } from '@chakra-ui/react'
import { Icon } from '@iconify/react'

type NodeType = 'photo' | 'video' | 'stream' | 'file' | 'task' | 'model' | 'preview' | 'save'

interface MenuItem {
  icon: string
  label: string
  type: NodeType
}

const MENU_ITEMS: MenuItem[] = [
  { icon: 'mdi:image', label: 'Источник фото', type: 'photo' },
  { icon: 'mdi:video', label: 'Источник видео', type: 'video' },
  { icon: 'mdi:webcam', label: 'Источник поток', type: 'stream' },
  { icon: 'mdi:brain', label: 'Модель', type: 'model' },
  { icon: 'mdi:eye', label: 'Предпросмотр', type: 'preview' },
  { icon: 'mdi:content-save', label: 'Сохранение результатов', type: 'save' },
  { icon: 'mdi:file-document', label: 'Файл', type: 'file' },
  { icon: 'mdi:cube', label: 'Задача', type: 'task' },
]

interface NodePanelProps {
  onAddNode: (type: NodeType) => void
}

export const NodePanel = ({ onAddNode }: NodePanelProps) => {
  return (
    <VStack pt="25px" width="200px" spacing={4}>
      <List spacing={3}>
        {MENU_ITEMS.map((item) => (
          <ListItem
            key={item.type}
            display="flex"
            alignItems="center"
            p={2}
            borderRadius="md"
            _hover={{ bg: 'gray.100' }}
            cursor="pointer"
            onClick={() => onAddNode(item.type)}
          >
            <IconButton
              aria-label={item.label}
              icon={<Icon icon={item.icon} width="24" height="24" />}
              variant="ghost"
              mr={2}
            />
            <Text>{item.label}</Text>
          </ListItem>
        ))}
      </List>
    </VStack>
  )
}

export default NodePanel 