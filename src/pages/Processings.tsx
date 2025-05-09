import { Box, Text, Table, Button } from '@chakra-ui/react'
import { useEffect, useState } from 'react'
import { getAllProcessings, IProcessing } from '../api/processing'
import dayjs from 'dayjs'
import { useNavigate } from 'react-router-dom'

const ProcessingsPage = () => {
    const [data, setData] = useState<null | IProcessing[]>(null)
    useEffect(() => {
        getAllProcessings().then(res => {
            setData(res?.data ? res.data : null)
        })
    }, [])

    const navigate = useNavigate()

    return (
        <Box p={4} pt={0} width='100%'>
            <Box display='flex' justifyContent='space-between'>
                <Text fontSize="2xl" mb={4}>Обработка</Text>
                <Button variant="surface" onClick={() => navigate('/pipeline')}>Новый пайплайн</Button>
            </Box>
            <Table.Root variant="outline">
                <Table.Header>
                    <Table.Row>
                        <Table.ColumnHeader>ID</Table.ColumnHeader>
                        <Table.ColumnHeader>Тип</Table.ColumnHeader>
                        <Table.ColumnHeader>Модель</Table.ColumnHeader>
                        <Table.ColumnHeader>Данные</Table.ColumnHeader>
                        <Table.ColumnHeader>Статус</Table.ColumnHeader>
                        <Table.ColumnHeader>Создан</Table.ColumnHeader>
                    </Table.Row>
                </Table.Header>
                <Table.Body>
                    {!data && (
                        <Table.Row>
                            <Table.Cell colSpan={5}>Ошибка при загрузке данных</Table.Cell>
                        </Table.Row>
                    )}
                    {data && data.length === 0 && (
                        <Table.Row>
                            <Table.Cell colSpan={5}>Нет данных</Table.Cell>
                        </Table.Row>
                    )}
                    {data && data.length > 0 && (
                        data.map((processing) => (
                            <Table.Row>
                                <Table.Cell>{processing.id || '-'}</Table.Cell>
                                <Table.Cell>{processing.type || '-'}</Table.Cell>
                                <Table.Cell>{
                                    // processing.model
                                    '-'}</Table.Cell>
                                <Table.Cell>{processing.rtsp_url || '-'}</Table.Cell>
                                <Table.Cell>{processing.status || '-'}</Table.Cell>
                                <Table.Cell>{dayjs(processing.created_at).format('HH:mm, DD.MM.YY')}</Table.Cell>
                            </Table.Row>
                        ))
                    )}
                </Table.Body>
            </Table.Root>
        </Box>
    )
}

export default ProcessingsPage