import {
  Box,
  createListCollection,
  IconButton,
  Input,
  Portal,
  Select,
  Text,
} from '@chakra-ui/react'
import { createProcessing } from '../shared/api/processing'
import { Button, ButtonGroup, Dialog } from '@chakra-ui/react'
import { useState } from 'react'
import ReactPlayer from 'react-player'
import { checkPipeline, stopCheckPipeline } from '../shared/api/processing'
import { useNavigate } from 'react-router-dom'
import { Icon as Iconify } from '@iconify/react'

const PipelinePage = () => {
  const frameworks = createListCollection({
    items: [{ label: 'RTSP', value: 'rtsp' }],
  })
  const models = createListCollection({
    items: [{ label: 'YOLO8', value: 'yolo8' }],
  })

  const [inputType, setInputType] = useState<string[]>([])
  const [rtspURL, setRtspURL] = useState('')
  const [selectedModel, setSelectedModel] = useState<string[]>([])
  const [triggerClass, setTriggerClass] = useState('')
  const [threshold, setThreshold] = useState('')

  const handleReset = () => {
    setInputType([])
    setRtspURL('')
    setSelectedModel([])
  }

  const [openVerify, setOpenVerify] = useState(false)
  const handleOpenVerify = () => {
    handleTestPipeline()
    setOpenVerify(true)
  }
  const handleCloseVerify = () => {
    setOpenVerify(false)
    stopCheckPipeline(pipelineID)
      .then((response) => {
        console.log(response)
      })
      .catch((e) => console.log(e))
  }

  const [flag, setFlag] = useState(0)
  const handleRefresh = () => {
    setFlag(Date.now())
  }

  const [pipelineID, setPipelineID] = useState('')
  const handleTestPipeline = () => {
    checkPipeline(rtspURL, selectedModel?.[0] || '')
      .then((response) => {
        console.log(response)
        setPipelineID(response.data.stream_id)
      })
      .catch((e) => console.log(e))
  }
  const navigate = useNavigate()

  const handleSubmitPipeline = () => {
    createProcessing({
      type: inputType?.[0],
      rtsp_url: rtspURL,
      model: selectedModel?.[0],
      trigger_class: triggerClass,
      confidence_threshold: threshold,
    }).then((res) => {
      if (res?.data?.task_id) {
        handleCloseVerify()
        handleReset()
        navigate('/processings')
      }
    })
  }

  return (
    <Box p={4} pt={0} width="100%">
      <Box display="flex" alignItems="center" mb={4}>
        <IconButton
          variant="ghost"
          mr={2}
          onClick={() => navigate('/processings')}
          size="xs"
        >
          <Iconify icon="mdi:chevron-left" />
        </IconButton>
        <Text fontSize="2xl">Создать пайплайн обработки</Text>
      </Box>
      <Box mb={2}>
        <Text mb={2}>Выберите с каким типом данных вы хотите работать</Text>
        <Select.Root
          collection={frameworks}
          size="sm"
          value={inputType}
          onValueChange={(e) => setInputType(e.value)}
        >
          <Select.HiddenSelect />
          <Select.Control>
            <Select.Trigger>
              <Select.ValueText placeholder="Выберите" />
            </Select.Trigger>
            <Select.IndicatorGroup>
              <Select.Indicator />
            </Select.IndicatorGroup>
          </Select.Control>
          <Portal>
            <Select.Positioner>
              <Select.Content>
                {frameworks.items.map((framework) => (
                  <Select.Item item={framework} key={framework.value}>
                    {framework.label}
                    <Select.ItemIndicator />
                  </Select.Item>
                ))}
              </Select.Content>
            </Select.Positioner>
          </Portal>
        </Select.Root>
      </Box>
      {Boolean(inputType?.[0]) && (
        <>
          <Box mb={2}>
            <Text mb={2}>Введите адрес потока</Text>
            <Input
              placeholder="rtsp://имя_пользователя:пароль@адрес_потока"
              value={rtspURL}
              onChange={(e) => setRtspURL(e.target.value)}
            />
          </Box>
          <Box mb={2}>
            <Text mb={2}>Выберите модель</Text>
            <Box display="flex">
              <Select.Root
                collection={models}
                size="sm"
                value={selectedModel}
                onValueChange={(e) => setSelectedModel(e.value)}
              >
                <Select.HiddenSelect />
                <Select.Control>
                  <Select.Trigger>
                    <Select.ValueText placeholder="Выберите" />
                  </Select.Trigger>
                  <Select.IndicatorGroup>
                    <Select.Indicator />
                  </Select.IndicatorGroup>
                </Select.Control>
                <Portal>
                  <Select.Positioner>
                    <Select.Content>
                      {models.items.map((model) => (
                        <Select.Item item={model} key={model.value}>
                          {model.label}
                          <Select.ItemIndicator />
                        </Select.Item>
                      ))}
                    </Select.Content>
                  </Select.Positioner>
                </Portal>
              </Select.Root>
            </Box>
          </Box>
          <Box mb={2}>
            <Text mb={2}>Введите класс тригер</Text>
            <Box display="flex">
              <Input
                placeholder="Класс тригер (truck, person)"
                value={triggerClass}
                onChange={(e) => setTriggerClass(e.target.value)}
              />
            </Box>
          </Box>
          <Box mb={2}>
            <Text mb={2}>Введите уверенность</Text>
            <Box display="flex">
              <Input
                placeholder="Увереность (0.5)"
                value={threshold}
                onChange={(e) => setThreshold(e.target.value)}
              />
            </Box>
          </Box>
          <Box>
            <ButtonGroup>
              <Button variant="outline" onClick={handleReset}>
                Сбросить
              </Button>
              <Button
                onClick={handleOpenVerify}
                disabled={Boolean(
                  !inputType?.[0] || !selectedModel?.[0] || !rtspURL
                )}
              >
                Далее
              </Button>
            </ButtonGroup>
          </Box>
        </>
      )}

      <Dialog.Root open={openVerify} closeOnInteractOutside={false}>
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content>
              <Dialog.Header>
                <Dialog.Title>Проверка</Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <Button onClick={handleRefresh} mb={2}>
                  Обновить
                </Button>
                <Box
                  width="100%"
                  height="200px"
                  border="1px"
                  borderColor="gray.200"
                  borderRadius="md"
                >
                  <ReactPlayer
                    key={flag}
                    url={`api/static/streams/${pipelineID}/hls/playlist.m3u8`}
                    width="100%"
                    height="100%"
                    controls
                    playing
                    muted
                    loop
                    config={{
                      file: {
                        attributes: {
                          controlsList: 'nodownload noremoteplayback',
                        },
                        forceHLS: true,
                      },
                    }}
                  />
                </Box>
              </Dialog.Body>
              <Dialog.Footer>
                <Dialog.ActionTrigger asChild>
                  <Button variant="outline" onClick={handleCloseVerify}>
                    Отмена
                  </Button>
                </Dialog.ActionTrigger>
                <Button onClick={handleSubmitPipeline}>
                  Запустить пайплайн
                </Button>
              </Dialog.Footer>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
    </Box>
  )
}

export default PipelinePage
