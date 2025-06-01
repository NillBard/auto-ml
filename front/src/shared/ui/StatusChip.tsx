import { Badge, ProgressCircle } from '@chakra-ui/react'
import { Icon } from '@iconify/react/dist/iconify.js'

interface StatusChipProps {
  status: string
}
const statusMap: Record<StatusChipProps['status'], string> = {
  pending: 'В ожидании',
  processing: 'В процессе',
  processed: 'Обработано',
  error: 'Ошибка',
}

export const StatusChip = ({ status }: StatusChipProps) => {
  let colorScheme: 'green' | 'yellow' | 'blue' | 'red' = 'blue'

  switch (status) {
    case 'pending':
      colorScheme = 'yellow'
      break
    case 'processing':
      colorScheme = 'blue'
      break
    case 'processed':
      colorScheme = 'green'
      break
    case 'error':
      colorScheme = 'red'
      break
  }

  return (
    <Badge colorPalette={colorScheme} p={1}>
      {statusMap[status]}
      {status === 'processing' && (
        <ProgressCircle.Root value={null} size="xs" ml={2}>
          <ProgressCircle.Circle css={{ '--thickness': '1px' }}>
            <ProgressCircle.Track />
            <ProgressCircle.Range />
          </ProgressCircle.Circle>
        </ProgressCircle.Root>
      )}
      {status === 'processed' && <Icon icon="mdi:check" width="24px" />}
      {status === 'error' && <Icon icon="mdi:error-outline" width="24px" />}
      {status === 'pending' && (
        <Icon icon="mdi:timer-sand-empty" width="24px" />
      )}
    </Badge>
  )
}
