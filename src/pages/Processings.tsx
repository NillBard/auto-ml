import { Box, Text, Table, Button, IconButton, Link } from '@chakra-ui/react'
import { useEffect, useState } from 'react'
import {
  downloadDetections,
  getAllProcessings,
  IProcessing,
} from '@/shared/api'
import dayjs from 'dayjs'
import { useNavigate } from 'react-router-dom'
import { Icon as Iconify } from '@iconify/react/dist/iconify.js'
import { StatusChip } from '@/shared/ui'

const ProcessingsPage = () => {
  const [data, setData] = useState<null | IProcessing[]>(null)
  useEffect(() => {
    getAllProcessings().then((res) => {
      setData(res?.data ? res.data : null)
    })
  }, [])

  const navigate = useNavigate()

  return (
    <Box p={4} pt={0} width="100%">
      <Box display="flex" justifyContent="space-between">
        <Text fontSize="2xl" mb={4}>
          Обработка
        </Text>
        <Button variant="surface" onClick={() => navigate('/pipeline')}>
          Новый пайплайн
        </Button>
      </Box>
      <Table.Root variant="outline">
        <Table.Header>
          <Table.Row>
            <Table.ColumnHeader>ID</Table.ColumnHeader>
            <Table.ColumnHeader>Тип</Table.ColumnHeader>
            <Table.ColumnHeader>Модель</Table.ColumnHeader>
            <Table.ColumnHeader>Триггер</Table.ColumnHeader>
            <Table.ColumnHeader>Данные</Table.ColumnHeader>
            <Table.ColumnHeader>Статус</Table.ColumnHeader>
            <Table.ColumnHeader>Создан</Table.ColumnHeader>
            <Table.ColumnHeader>Обнаружения</Table.ColumnHeader>
          </Table.Row>
        </Table.Header>
        <Table.Body>
          {!data && (
            <Table.Row>
              <Table.Cell colSpan={5}>Загрузка данных или ошибка</Table.Cell>
            </Table.Row>
          )}
          {data && data.length === 0 && (
            <Table.Row>
              <Table.Cell colSpan={5}>Нет данных</Table.Cell>
            </Table.Row>
          )}
          {data &&
            data.length > 0 &&
            data.map((processing) => (
              <Table.Row>
                <Table.Cell>{processing.id || '-'}</Table.Cell>
                <Table.Cell>{processing.type || '-'}</Table.Cell>
                <Table.Cell>{processing.model || '-'}</Table.Cell>
                <Table.Cell>{`${processing.trigger_class} ${processing.confidence_threshold}`}</Table.Cell>
                <Table.Cell>{processing.rtsp_url || '-'}</Table.Cell>
                <Table.Cell>
                  <StatusChip status={processing.status} />
                </Table.Cell>
                <Table.Cell>
                  {dayjs(processing.created_at).format('HH:mm, DD.MM.YY')}
                </Table.Cell>
                <Table.Cell
                  display="flex"
                  alignItems="center"
                  justifyContent="center"
                >
                  {processing.status === 'processed' && (
                    <IconButton size="xs" asChild>
                      <Link
                        download
                        href={`/api/processing/download/${processing.id}`}
                      >
                        <Iconify icon="mdi:download" />
                      </Link>
                    </IconButton>
                  )}
                </Table.Cell>
              </Table.Row>
            ))}
        </Table.Body>
      </Table.Root>
    </Box>
  )
}

export default ProcessingsPage
