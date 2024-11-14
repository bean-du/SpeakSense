# ASR Service API Documentation

## Authentication

所有接口都需要在请求头中包含 `Authorization` 字段，值为有效的 API Key。

## API Keys Management

### 1. 创建 API Key
**POST** `/auth/api-keys`

请求参数：
| 字段 | 类型 | 描述 |
|------|------|------|
| name | String | API Key 的名称 |
| permissions | Array\<Permission\> | 权限列表 |
| rate_limit | RateLimit | 速率限制配置 |
| expires_in_days | Option\<i64\> | 过期天数，可选 |

**Permission 枚举值：**
| 值 | 描述 |
|------|------|
| Transcribe | 转写权限 |
| SpeakerDiarization | 说话人分离权限 |
| Admin | 管理员权限 |

**RateLimit 结构体：**
| 字段 | 类型 | 描述 |
|------|------|------|
| requests_per_minute | u32 | 每分钟请求限制 |
| requests_per_hour | u32 | 每小时请求限制 |
| requests_per_day | u32 | 每天请求限制 |

### 2. 撤销 API Key
**DELETE** `/auth/api-keys/:api_key`

路径参数：
| 参数 | 描述 |
|------|------|
| api_key | API Key 字符串 |

### 3. 获取 API Key 统计信息
**GET** `/auth/api-keys/:api_key/stats`

路径参数：
| 参数 | 描述 |
|------|------|
| api_key | API Key 字符串 |

### 4. 获取 API Key 使用报告
**GET** `/auth/api-keys/:api_key/usage`

路径参数：
| 参数 | 描述 |
|------|------|
| api_key | API Key 字符串 |

## ASR Service

### 1. 音频转写
**POST** `/asr/transcribe`

请求参数：
| 字段 | 类型 | 描述 |
|------|------|------|
| audio_url | String | 音频文件 URL |
| callback_url | String | 回调 URL |
| language | Option\<String\> | 语言代码，可选 |
| speaker_diarization | bool | 是否启用说话人分离 |
| emotion_recognition | bool | 是否启用情感识别 |
| filter_dirty_words | bool | 是否过滤敏感词 |

## Task Management

### 1. 创建任务
**POST** `/schedule/tasks`

请求参数：
| 字段 | 类型 | 描述 |
|------|------|------|
| task_type | TaskType | 任务类型 |
| input_path | PathBuf | 输入文件路径 |
| callback_type | CallbackType | 回调类型 |
| params | TaskParams | 任务参数 |
| priority | TaskPriority | 任务优先级 |
| retry_count | u32 | 当前重试次数 |
| max_retries | u32 | 最大重试次数 |
| timeout | Option\<Duration\> | 超时时间 |

**TaskType 枚举值：**
| 值 | 描述 |
|------|------|
| Transcribe | 转写任务 |

**CallbackType 枚举值：**
| 值 | 描述 |
|------|------|
| Http { url: String } | HTTP 回调 |

**TaskParams 枚举值：**
| 值 | 字段 | 类型 | 描述 |
|------|------|------|------|
| Transcribe | language | Option\<String\> | 语言代码 |
| | speaker_diarization | bool | 是否启用说话人分离 |
| | emotion_recognition | bool | 是否启用情感识别 |
| | filter_dirty_words | bool | 是否过滤敏感词 |

**TaskPriority 枚举值：**
| 值 | 描述 |
|------|------|
| High | 高优先级 |
| Normal | 普通优先级 |
| Low | 低优先级 |

### 2. 获取任务信息
**GET** `/schedule/tasks/:task_id`

路径参数：
| 参数 | 描述 |
|------|------|
| task_id | 任务 ID |

### 3. 获取任务状态
**GET** `/schedule/tasks/:task_id/status`

路径参数：
| 参数 | 描述 |
|------|------|
| task_id | 任务 ID |

### 4. 更新任务优先级
**POST** `/schedule/tasks/:task_id/priority`

路径参数：
| 参数 | 描述 |
|------|------|
| task_id | 任务 ID |

请求参数：
| 字段 | 类型 | 描述 |
|------|------|------|
| priority | TaskPriority | 新的任务优先级 |

### 5. 获取任务统计信息
**GET** `/schedule/tasks/stats`

查询参数：
| 参数 | 类型 | 描述 |
|------|------|------|
| page | u32 | 页码 |
| page_size | u32 | 每页数量 