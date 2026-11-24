"""
知识库管理 API 路由
职责：知识库的创建、删除、列表查询、文档上传、内容检索
原则：只做协议转换，业务逻辑全部下沉到 KnowledgeBaseManager
"""
import os
import tempfile
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from api.schemas import (
    ApiResponse,
    KnowledgeBaseListResponse,
    KnowledgeBaseInfo,
    SearchKbRequest,
    SearchKbResponse,
    SearchResultItem,
    UploadDocumentResponse,
)
from api.depends import get_kb_manager
from rag_knowledge.kb_manager import KnowledgeBaseManager
from utils.logger_tool import logger

# 路由前缀统一
router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["知识库管理"])


# ===================== 1. 获取知识库列表 =====================
@router.get("", response_model=ApiResponse)
def list_knowledge_bases(
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager),
):
    """获取所有知识库列表及文档数量"""
    try:
        kb_names = kb_manager.list_kbs()
        items = []
        for name in kb_names:
            store = kb_manager._get_store(name)
            items.append(KnowledgeBaseInfo(
                name=name,
                doc_count=store.count()
            ))

        return ApiResponse(
            code=0,
            message="success",
            data=KnowledgeBaseListResponse(
                total=len(items),
                items=items
            )
        )
    except Exception as e:
        logger.error(f"获取知识库列表失败：{e}")
        raise HTTPException(status_code=500, detail=f"获取列表失败：{str(e)}")


# ===================== 2. 创建知识库 =====================
@router.post("/{kb_name}", response_model=ApiResponse)
def create_knowledge_base(
    kb_name: str,
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager),
):
    """
    创建一个新的知识库
    - kb_name: 知识库名称，英文/数字/下划线，作为唯一标识
    """
    # 简单合法性校验
    if not kb_name or len(kb_name) > 50 or not kb_name.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="知识库名称不合法，只能包含字母、数字、下划线，长度不超过50")

    try:
        # 懒加载机制：第一次访问就自动创建，这里主动触发一次
        kb_manager._get_store(kb_name)
        logger.info(f"创建知识库：{kb_name}")
        return ApiResponse(
            code=0,
            message=f"知识库 {kb_name} 创建成功",
            data={"name": kb_name}
        )
    except Exception as e:
        logger.error(f"创建知识库失败：{e}")
        raise HTTPException(status_code=500, detail=f"创建失败：{str(e)}")


# ===================== 3. 删除知识库 =====================
@router.delete("/{kb_name}", response_model=ApiResponse)
def delete_knowledge_base(
    kb_name: str,
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager),
):
    """删除指定知识库及所有文档"""
    try:
        success = kb_manager.delete_kb(kb_name)
        if not success:
            raise HTTPException(status_code=404, detail=f"知识库 {kb_name} 不存在")

        return ApiResponse(
            code=0,
            message=f"知识库 {kb_name} 删除成功",
            data=None
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除知识库失败：{e}")
        raise HTTPException(status_code=500, detail=f"删除失败：{str(e)}")


# ===================== 4. 上传文档到知识库 =====================
@router.post("/{kb_name}/upload", response_model=ApiResponse)
def upload_document(
    kb_name: str,
    file: UploadFile = File(..., description="上传的文档文件，支持txt/md/pdf"),
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager),
):
    """
    向指定知识库上传文档文件
    支持格式：.txt, .md, .pdf
    """
    # 校验知识库是否存在（通过尝试获取来判断）
    kb_list = kb_manager.list_kbs()
    if kb_name not in kb_list:
        raise HTTPException(status_code=404, detail=f"知识库 {kb_name} 不存在，请先创建")

    # 校验文件后缀
    allowed_exts = {".txt", ".md", ".markdown", ".pdf"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式 {file_ext}，支持：{', '.join(allowed_exts)}"
        )

    try:
        # 写入临时文件
        with tempfile.NamedTemporaryFile(
                suffix=file_ext,
                delete=False
        ) as tmp:
            # 分块读取写入，避免大文件占满内存
            content = file.file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 调用知识库管理器添加文档
        chunk_count = kb_manager.add_document(kb_name, tmp_path)

        # 清理临时文件
        os.unlink(tmp_path)

        if chunk_count == 0:
            raise HTTPException(status_code=400, detail="文档解析失败或内容为空")

        logger.info(f"知识库[{kb_name}] 上传文档成功：{file.filename}，分片数：{chunk_count}")

        return ApiResponse(
            code=0,
            message="上传成功",
            data=UploadDocumentResponse(
                kb_name=kb_name,
                filename=file.filename,
                chunk_count=chunk_count
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传文档失败：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"上传失败：{str(e)}")


# ===================== 5. 知识库检索（测试用） =====================
@router.post("/{kb_name}/search", response_model=ApiResponse)
def search_knowledge_base(
    kb_name: str,
    request: SearchKbRequest,
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager),
):
    """
    手动检索知识库，用于测试检索效果
    正式使用时通过 Agent 工具调用，不直接调用此接口
    """
    kb_list = kb_manager.list_kbs()
    if kb_name not in kb_list:
        raise HTTPException(status_code=404, detail=f"知识库 {kb_name} 不存在")

    try:
        results = kb_manager.search(
            kb_name=kb_name,
            query=request.query,
            top_k=request.top_k
        )

        formatted = [
            SearchResultItem(
                content=item["content"],
                source=item["source"],
                score=item["score"]
            )
            for item in results
        ]

        return ApiResponse(
            code=0,
            message="success",
            data=SearchKbResponse(
                kb_name=kb_name,
                results=formatted
            )
        )
    except Exception as e:
        logger.error(f"检索知识库失败：{e}")
        raise HTTPException(status_code=500, detail=f"检索失败：{str(e)}")
