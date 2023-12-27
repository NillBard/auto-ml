import ReactDOM from 'react-dom/client'
import { ChakraProvider } from '@chakra-ui/react'

import CreatePipeline from './pages/createPipeline'
import { ReactFlowProvider } from 'reactflow'

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <ChakraProvider>
    <ReactFlowProvider>
      <CreatePipeline />
    </ReactFlowProvider>
  </ChakraProvider>
)
