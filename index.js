const { Client } = require('eris')
const Groq = require('groq-sdk')

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY })

// Models to try in order — all free on Groq
const TEXT_MODELS = [
    'llama-3.3-70b-versatile',
    'llama-3.1-8b-instant',
    'gemma2-9b-it',
]

// Vision-capable models
const VISION_MODELS = [
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'meta-llama/llama-4-maverick-17b-128e-instruct',
]

const SYSTEM_PROMPT = 'You are a helpful, friendly AI assistant chatting via Discord. Keep responses concise and conversational. Use Discord markdown formatting when appropriate.'

// Per-user conversation history
const histories = new Map()

// Deduplication — prevent processing the same message twice
const processedMessages = new Set()

// Per-user lock — prevent concurrent AI calls for the same user
const processingUsers = new Set()

function getHistory(userId) {
    if (!histories.has(userId)) histories.set(userId, [])
    return histories.get(userId)
}

function trimHistory(userId) {
    const h = histories.get(userId)
    if (h && h.length > 20) histories.set(userId, h.slice(-20))
}

async function askAI(userId, text, imageUrls = []) {
    const history = getHistory(userId)
    const hasImages = imageUrls.length > 0

    // Build user content
    let userContent
    if (hasImages) {
        userContent = [
            { type: 'text', text: text || 'What is in this image?' },
            ...imageUrls.map((url) => ({ type: 'image_url', image_url: { url } })),
        ]
    } else {
        userContent = text
    }

    // Store plain text in history
    history.push({ role: 'user', content: text || '(sent an image)' })

    const messages = [
        { role: 'system', content: SYSTEM_PROMPT },
        ...history.slice(0, -1),
        { role: 'user', content: userContent },
    ]

    const modelsToTry = hasImages
        ? [...new Set([...VISION_MODELS, ...TEXT_MODELS])]
        : TEXT_MODELS

    let lastError
    for (const model of modelsToTry) {
        try {
            const response = await groq.chat.completions.create({ model, messages })
            const reply = response.choices[0].message.content
            history.push({ role: 'assistant', content: reply })
            trimHistory(userId)
            return reply
        } catch (err) {
            console.warn(`Model ${model} failed: ${err.message} — trying next...`)
            lastError = err
        }
    }

    history.pop()
    throw lastError
}

function getImageUrls(msg) {
    const imageTypes = ['image/png', 'image/jpeg', 'image/gif', 'image/webp']
    return (msg.attachments || [])
        .filter((a) => imageTypes.includes(a.content_type) || /\.(png|jpe?g|gif|webp)$/i.test(a.filename))
        .map((a) => a.url)
}

async function sendReply(channel, text) {
    const chunks = text.match(/[\s\S]{1,2000}/g) || []
    for (const chunk of chunks) {
        await channel.createMessage({ content: chunk, flags: 4 }) // flags: 4 = SUPPRESS_EMBEDS
    }
}

async function init(token) {
    const bot = new Client(`Bot ${token}`, {
        intents: ['guilds', 'directMessages', 'guildMessages', 'messageContent'],
        maxShards: 'auto',
        restMode: true,
    })

    bot.on('ready', async () => {
        await bot.bulkEditCommands([
            {
                name: 'chat',
                description: 'Chat with the AI assistant',
                type: 1,
                options: [{ name: 'message', description: 'Your message to the AI', type: 3, required: true }],
            },
            {
                name: 'reset',
                description: 'Reset your conversation history with the AI',
                type: 1,
            },
        ])
        console.log(`Bot ready! Logged in as ${bot.user.username}`)
        console.log(`Invite URL: https://discord.com/oauth2/authorize?client_id=${bot.user.id}&scope=applications.commands%20bot&permissions=3072`)
        console.log('Listening for DMs, mentions, and /chat commands')
    })

    bot.on('interactionCreate', async (interaction) => {
        if (interaction.type !== 2) return

        if (interaction.data.name === 'chat') {
            const userMessage = interaction.data.options[0].value
            await interaction.acknowledge()
            try {
                const reply = await askAI(interaction.member?.id || interaction.user.id, userMessage)
                await interaction.createFollowup({ content: reply, flags: 4 })
            } catch (err) {
                console.error('AI error:', err.message)
                await interaction.createFollowup({ content: 'Sorry, something went wrong. Try again in a moment!' })
            }
        }

        if (interaction.data.name === 'reset') {
            const userId = interaction.member?.id || interaction.user.id
            histories.delete(userId)
            await interaction.createMessage({ content: 'Conversation history cleared!' })
        }
    })

    bot.on('messageCreate', async (msg) => {
        if (msg.author.bot) return
        console.log(`[messageCreate] id=${msg.id} author=${msg.author.id} guild=${msg.guildID || 'DM'} already=${processedMessages.has(msg.id)} processing=${processingUsers.has(msg.author.id)}`)
        if (processedMessages.has(msg.id)) return
        processedMessages.add(msg.id)
        // Clean up old IDs to avoid memory leak
        if (processedMessages.size > 500) {
            const first = processedMessages.values().next().value
            processedMessages.delete(first)
        }

        const isDM = !msg.guildID
        const isMention = msg.mentions.some((u) => u.id === bot.user.id)
        if (!isDM && !isMention) return

        // Only one response at a time per user
        if (processingUsers.has(msg.author.id)) return
        processingUsers.add(msg.author.id)

        const text = msg.content
            .replace(`<@${bot.user.id}>`, '')
            .replace(`<@!${bot.user.id}>`, '')
            .trim()

        const imageUrls = getImageUrls(msg)

        if (!text && imageUrls.length === 0) {
            processingUsers.delete(msg.author.id)
            await msg.channel.createMessage('Hey! What can I help you with?')
            return
        }

        try {
            await msg.channel.sendTyping()
            const reply = await askAI(msg.author.id, text, imageUrls)
            await sendReply(msg.channel, reply)
        } catch (err) {
            console.error('AI error:', err.message)
            await msg.channel.createMessage('Sorry, something went wrong. Try again in a moment!')
        } finally {
            processingUsers.delete(msg.author.id)
        }
    })

    bot.connect()
}

const token = process.argv[2] || process.env.BOT_TOKEN
if (!token) {
    console.error('No bot token provided. Set the BOT_TOKEN environment secret or pass it as an argument.')
    process.exit(1)
}

init(token)
